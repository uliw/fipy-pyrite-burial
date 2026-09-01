"""Build the equation matrix and call the respective solvers."""

from __future__ import annotations

import gc
import math
import time
import traceback
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Callable

from fipy import CellVariable
from fipy.terms.diffusionTerm import DiffusionTerm
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
from fipy.terms.transientTerm import TransientTerm

# import numpy as np
from fipy.tools import numerix as np

from .diff_lib import (
    get_time_units,
    save_data,
    save_data_async,
    save_state,
)
from .live_plot_lib import write_to_queue_async

if TYPE_CHECKING:
    from fipy.meshes.mesh import Mesh


@dataclass
class AdaptiveDT:
    """
    Advanced PID-controlled adaptive time stepping.

    Uses standard H211B controller logic for smooth step size adaptation.
    """

    dt_min: float
    dt_max: float
    dt_initial: float
    dt_cfl_factor: float = 0.8
    growth_factor: float = 1.2
    cut_factor: float = 0.5
    pid: bool = True
    kP: float = 0.075
    kI: float = 0.175
    kD: float = 0.01

    _dt: float = 0.0
    _err_prev: float | None = None  # Delay initialization
    _dt_prev: float = 0.0

    def __post_init__(self) -> None:
        self._dt = max(self.dt_min, min(self.dt_initial, self.dt_max))
        self._dt_prev = self._dt

    @property
    def dt(self) -> float:
        """Return current time step."""
        return self._dt

    def cfl_limit(self, dx: float, vel: float, D: float) -> float:
        """Compute global CFL estimate for advection-diffusion."""
        adv_limit = math.inf if vel == 0 else dx / abs(vel)
        dif_limit = math.inf if D == 0 else dx * dx / (2 * D)
        return self.dt_cfl_factor * min(adv_limit, dif_limit)

    def update(
        self,
        error_metric: float,
        dt_cfl: float | None = None,
        step_success: bool = True,
        target_error: float = 1e-4,
    ) -> float:
        """
        Compute the next dt based on solver performance and change magnitude.

        Parameters:
        -----------
        error_metric : float
            Magnitude of variable change (e.g., Max or RMS) used for control.
        dt_cfl : float, optional
            Hard upper bound based on stability limits.
        step_success : bool
            Whether the linear solver converged.
        target_error : float
            The desired change per step.
        """
        # 1. Handle step failure
        if not step_success:
            self._dt = max(self._dt * self.cut_factor, self.dt_min)
            return self._dt

        # 2. PID Control (H211B)
        if self.pid:
            # We want max_change to be around target_error
            # If this is the first step, assume we are at the target
            if self._err_prev is None:
                self._err_prev = target_error

            err = max(error_metric, 1e-25)
            err_prev = max(self._err_prev, 1e-25)
            err_ref = target_error

            # PID Factor calculation: (ref/err) makes it shrink if err > ref
            factor = (
                (err_prev / err) ** self.kP
                * (err_ref / err) ** self.kI
                * (err_ref / err_prev) ** self.kD
            )

            # Dampen and limit the factor
            factor = max(self.cut_factor, min(self.growth_factor, factor))
            self._dt = max(self.dt_min, min(self._dt * factor, self.dt_max))
        else:
            # Simplistic growth/cut logic
            if error_metric < target_error:
                self._dt = min(self._dt * self.growth_factor, self.dt_max)
            else:
                self._dt = max(self._dt * self.cut_factor, self.dt_min)

        # 3. Apply CFL cap if provided
        if dt_cfl is not None:
            self._dt = min(self._dt, dt_cfl)

        # 4. Store state
        self._err_prev = error_metric
        return self._dt


def _get_solver(mp: Any) -> Any:
    """Initialize and return the FiPy solver based on configuration."""
    backend = mp.solver_backend
    tol = mp.tolerance
    if getattr(mp, "isotopes", False):
        tol = min(tol, 1e-10)

    if backend == "default":
        from fipy import DefaultSolver

        solver = DefaultSolver(tolerance=tol)
    elif backend == "LinearLUSolver":
        from fipy import LinearLUSolver

        solver = LinearLUSolver(tolerance=tol)
    else:
        from petsc4py import PETSc
        if getattr(mp, "solver_monitor", False):
            PETSc.Options().setValue("ksp_monitor", "")
            PETSc.Options().setValue("ksp_converged_reason", "")

        # PETSc version-specific fix for converged reason constants
        if not hasattr(PETSc.KSP.ConvergedReason, "CONVERGED_ATOL_NORMAL"):
            PETSc.KSP.ConvergedReason.CONVERGED_ATOL_NORMAL = (
                PETSc.KSP.ConvergedReason.CONVERGED_ATOL_NORMAL_EQUATIONS
            )
        if not hasattr(PETSc.KSP.ConvergedReason, "CONVERGED_RTOL_NORMAL"):
            PETSc.KSP.ConvergedReason.CONVERGED_RTOL_NORMAL = (
                PETSc.KSP.ConvergedReason.CONVERGED_RTOL_NORMAL_EQUATIONS
            )

        if backend == "LinearGMRESSolver":
            from fipy.solvers.petsc import LinearGMRESSolver

            precon = getattr(mp, "solver_precon", "hypre")
            solver_kwargs = {"precon": precon, "tolerance": tol}
            if hasattr(mp, "solver_atol") and mp.solver_atol is not None:
                solver_kwargs["absolute_tolerance"] = mp.solver_atol
            if hasattr(mp, "solver_max_iterations") and mp.solver_max_iterations is not None:
                solver_kwargs["iterations"] = mp.solver_max_iterations

            solver = LinearGMRESSolver(**solver_kwargs)
        elif backend == "petscSolver":
            # this is currently not working
            from fipy.solvers.petsc import petscSolver

            solver = petscSolver(tolerance=tol)

        elif backend == "PETScNewtonSolver":
            raise ValueError(
                "PETScNewtonSolver is not a valid FiPy solver backend. "
                "FiPy does not have a native PETSc Newton solver class. "
                "Please use 'LinearGMRESSolver' or 'LinearLUSolver' instead, "
                "and handle nonlinearities through sweeping."
            )

    return solver


def _build_passive_eqs(
    mp: Any,
    c: Any,
    mesh: Mesh,
    D_mol: Any,
    bc_map: Dict[str, Any],
    species_list_partial: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build the invariant parts of the species equations (transport and transient).

    Returns:
    --------
    species_struct : list of dicts containing variable references
    passive_eqs : dict mapping species names to their partial FiPy equations
    """
    species_struct = []
    passive_eqs = {}

    # Ensure mp.phi is a CellVariable even if provided as a float
    # We create it once outside the loop to avoid recreating it for each species
    if not isinstance(mp.phi, CellVariable):
        phi_var = CellVariable(mesh=mesh, value=mp.phi)
        mp.phi = phi_var
    else:
        phi_var = mp.phi

    for name in species_list_partial:
        var = getattr(c, name)
        props = bc_map[name]

        species_struct.append({"name": name, "var": var})

        # Effective porosity for conservative form
        eff_phi = phi_var if props["type"] == "dissolved" else (1.0 - phi_var)

        # Diffusion coefficient (Molecular + Bio-diffusion)
        D_total = np.maximum(getattr(D_mol, name) + D_mol.D_bio, 1e-20)

        # Advection velocity
        vel = getattr(mp, "w", 0.0) - getattr(mp, "advection", 0.0) if props["type"] == "dissolved" else getattr(mp, "w", 0.0)
        u_var = CellVariable(mesh=mesh, value=vel, rank=1)

        # Terms with conservative phi handling
        conv_term = PowerLawConvectionTerm(coeff=eff_phi * u_var, var=var)
        diff_term = DiffusionTerm(
            coeff=eff_phi * CellVariable(mesh=mesh, value=D_total), var=var
        )

        # Irrigation (Sources/Sinks for dissolved species)
        irr_term = 0.0
        if props["type"] == "dissolved":
            irr_term = ImplicitSourceTerm(
                coeff=eff_phi * CellVariable(mesh=mesh, value=-D_mol.D_irr), var=var
            ) + eff_phi * CellVariable(mesh=mesh, value=D_mol.D_irr * props["top"])

        # Passive equation: Transient + Convection - Diffusion - Irrigation
        passive_eqs[name] = (
            TransientTerm(coeff=eff_phi, var=var) + conv_term - diff_term - irr_term
        )

    return species_struct, passive_eqs


def _setup_static_coupled_equation(
    mp: Any,
    c: Any,
    k: Any,
    mesh: Mesh,
    passive_eqs: Dict[str, Any],
    species_struct: List[Dict[str, Any]],
    diagenetic_reactions: Any,
    species_list_partial: List[str],
) -> Tuple[
    Any, Dict[str, CellVariable], Dict[str, CellVariable], Dict[str, List[CellVariable]]
]:
    """
    Setup static coefficient variables and compile the coupled equation system once.
    """
    from .diff_lib import data_container

    f_dummy = data_container()

    # Discover the coupling structure using a dummy run
    mp.in_solver = True
    try:
        f_res, _ = diagenetic_reactions(mp, c, k, f=f_dummy)
    finally:
        mp.in_solver = False

    LHS_vars = {}
    RHS_vars = {}
    CROSS_vars = {}  # species -> list of CellVariable

    eqs = []
    for s_obj in species_struct:
        name = s_obj["name"]

        # Pre-allocate static variables
        LHS_vars[name] = CellVariable(mesh=mesh, value=0.0)
        RHS_vars[name] = CellVariable(mesh=mesh, value=0.0)
        CROSS_vars[name] = []

        # Build coupled off-diagonal terms
        cross_list = f_res.raw_CROSS.get(name, [])
        cross_term = 0.0
        for source_name, _ in cross_list:
            v_cross = CellVariable(mesh=mesh, value=0.0)
            CROSS_vars[name].append(v_cross)
            cross_term += ImplicitSourceTerm(coeff=v_cross, var=c[source_name])

        lhs_reaction = ImplicitSourceTerm(coeff=LHS_vars[name], var=s_obj["var"])
        eq = passive_eqs[name] == lhs_reaction + cross_term + RHS_vars[name]
        eqs.append(eq)

    coupled_eq = reduce(lambda a, b: a & b, eqs)
    return coupled_eq, LHS_vars, RHS_vars, CROSS_vars


def _update_static_coefficients(
    mp: Any,
    c: Any,
    k: Any,
    diagenetic_reactions: Any,
    LHS_vars: Dict[str, CellVariable],
    RHS_vars: Dict[str, CellVariable],
    CROSS_vars: Dict[str, List[CellVariable]],
    species_list_partial: List[str],
) -> Dict[str, np.ndarray]:
    """
    Calculate new reaction rates and update the static coefficient variables in-place.
    """
    from .diff_lib import data_container

    class ArrayProxy:
        def __init__(self, val):
            self.value = val
        def __getattr__(self, name):
            return getattr(self.value, name)
        def __add__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(self.value + val)
        def __radd__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(val + self.value)
        def __sub__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(self.value - val)
        def __rsub__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(val - self.value)
        def __mul__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(self.value * val)
        def __rmul__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(val * self.value)
        def __truediv__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(self.value / val)
        def __rtruediv__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(val / self.value)
        def __pow__(self, power):
            return ArrayProxy(self.value ** power)
        def __neg__(self):
            return ArrayProxy(-self.value)
        def __getitem__(self, idx):
            return self.value[idx]

    c_numpy = data_container({s: ArrayProxy(val.value) for s, val in c.items()})
    mp_numpy = data_container(mp)
    mp_numpy.phi = ArrayProxy(mp.phi.value)

    f_res = data_container()

    mp_numpy.in_solver = True
    try:
        f_res, RATES = diagenetic_reactions(mp_numpy, c_numpy, k, f=f_res)
    finally:
        mp_numpy.in_solver = False

    def get_val(val):
        if hasattr(val, "value"):
            return val.value
        return val

    for s in species_list_partial:
        # Update diagonal coefficient
        LHS_vars[s].setValue(get_val(f_res.raw_LHS[s]))
        # Update explicit source
        RHS_vars[s].setValue(get_val(f_res.raw_RHS[s]))
        # Update off-diagonal couplings
        cross_list = f_res.raw_CROSS[s]
        for v_cross, (source_name, coeff) in zip(CROSS_vars[s], cross_list):
            v_cross.setValue(get_val(coeff))
    RATES_numpy = {key: get_val(val) for key, val in RATES.items()}
    return RATES_numpy


def _calculate_dt_max_isotope(mp: Any, k: Any, D_mol: Any) -> float:
    """Calculates the maximum timestep constraint for isotope coupling dynamically
    based on the pseudo-first-order kinetic relaxation rate k_eff.

    Formula:
        dt_max_isotope = min(dt_max_user, gamma / k_eff)
    where gamma = 200.0 is the dimensionless kinetic coupling factor.
    """
    dt_max_user = float(getattr(mp, "dt_max_isotope", getattr(mp, "dt_max", 604800.0)))
    
    isotope_limiter_species = getattr(mp, "isotope_limiter_species", "FeS")
    if isotope_limiter_species == "FeS" and hasattr(k, "FeS_isp"):
        k_prec = float(getattr(k, "FeS_isp", 0.0))
        Hplus = float(getattr(k, "Hplus", 3.162e-5))
        FeS_sp = float(getattr(k, "FeS_sp", 0.3162))
        hs_frac = float(getattr(mp, "hs_frac", 0.76))
        
        bc_Fe3 = float(getattr(mp, "bc_Fe3", 0.0))
        w = max(float(getattr(mp, "w", 0.0)), 1e-20)
        phi_val = getattr(mp, "phi", 0.8)
        if hasattr(phi_val, "value"):
            phi_val = phi_val.value[0] if hasattr(phi_val.value, "__getitem__") else float(phi_val.value)
        phi_val = float(phi_val)
        Fe2_diss = float(getattr(mp, "Fe2_diss", 1.0))
        
        if bc_Fe3 > 0.0:
            Fe2_pw = (bc_Fe3 * Fe2_diss) / (w * phi_val)
        else:
            Fe2_pw = 0.1
            
        omega_den = Hplus * FeS_sp + 1e-30
        dO_dTS2 = Fe2_pw * hs_frac / omega_den
        k_eff = k_prec * 2.0 * dO_dTS2
        
        if k_eff > 0:
            gamma = float(getattr(mp, "isotope_gamma", 200.0))
            dt_kinetic = gamma / k_eff
            return float(min(dt_max_user, dt_kinetic))

    return dt_max_user


def _validate_rates(
    monitored_rate_species: List[str],
    RATES_tentative: Dict[str, np.ndarray],
    prev_rates: Dict[str, np.ndarray],
    prev_rates_2: Dict[str, np.ndarray],
    current_dt: float,
    prev_dt: float,
    prev_dt_2: float,
    rate_threshold: float,
    enable_rate_magnitude_check: bool,
) -> Tuple[bool, str]:
    """Performs rate validation checks (consecutive sign changes and magnitude checks)."""
    violation = False
    violation_reason = ""
    for name in monitored_rate_species:
        if name not in RATES_tentative or name not in prev_rates:
            continue
        
        r_tentative = np.asarray(RATES_tentative[name])
        r_prev = np.asarray(prev_rates[name])

        c_change_tentative = r_tentative * current_dt
        c_change_prev = r_prev * prev_dt
        
        # 1. Sign change check
        mask_sign = (np.abs(c_change_prev) >= rate_threshold) & (np.abs(c_change_tentative) >= rate_threshold)
        if np.any(mask_sign):
            abs_change = np.abs(c_change_tentative - c_change_prev)
            flipped_tentative = (c_change_tentative * c_change_prev < 0) & (abs_change >= 5.0 * rate_threshold)
            
            if name in prev_rates_2:
                r_prev_2 = np.asarray(prev_rates_2[name])
                c_change_prev_2 = r_prev_2 * prev_dt_2
                mask_sign_prev = (np.abs(c_change_prev_2) >= rate_threshold) & (np.abs(c_change_prev) >= rate_threshold)
                abs_change_prev = np.abs(c_change_prev - c_change_prev_2)
                flipped_prev = (c_change_prev * c_change_prev_2 < 0) & (abs_change_prev >= 5.0 * rate_threshold)
                osc_mask = mask_sign & mask_sign_prev & flipped_tentative & flipped_prev
            else:
                osc_mask = np.zeros_like(mask_sign, dtype=bool)

            if np.any(osc_mask):
                idx = np.where(osc_mask)[0][0]
                violation = True
                r_prev_2 = np.asarray(prev_rates_2[name])
                violation_reason = (
                    f"Consecutive sign changes (oscillation) in {name} rate at cell {idx} "
                    f"(prev2 rate: {r_prev_2[idx]:.2e}, prev rate: {r_prev[idx]:.2e}, tentative rate: {r_tentative[idx]:.2e}, conc_change: {abs_change[idx]:.2e} mmol/L)"
                )
                break
        
        # 2. Order-of-magnitude check
        if enable_rate_magnitude_check:
            mask_magnitude = (np.abs(c_change_prev) >= rate_threshold) & (np.abs(r_prev) >= 1e-8)
            if np.any(mask_magnitude):
                ratio = np.abs(c_change_tentative[mask_magnitude]) / np.abs(c_change_prev[mask_magnitude])
                large_increase = ratio > 10.0
                if np.any(large_increase):
                    idx = np.where(mask_magnitude)[0][np.where(large_increase)[0][0]]
                    violation = True
                    violation_reason = (
                        f"Order-of-magnitude rate increase in {name} at cell {idx} "
                        f"(prev: {r_prev[idx]:.2e}, tentative: {r_tentative[idx]:.2e}, ratio: {ratio[large_increase][0]:.2f})"
                    )
                    break
                    
    return violation, violation_reason


def _report_step_status(
    step: int,
    total_time: float,
    current_dt: float,
    rms_change: float,
    mp: Any,
    c: Any,
    z: np.ndarray,
    species_list_full: List[str],
    D_mol: Any,
    diagenetic_reactions: Any,
    equilibrium_reactions: Any,
    plot_queue: Optional[Any],
    _log: Callable[[str], None],
) -> None:
    """Logs current step status parameters and triggers async data/plot saving."""
    from .diff_lib import get_delta, get_total_delta
    
    phi = mp.phi
    dz = np.diff(z)
    fe_total_bulk = phi * c.Fe2_total + (1 - phi) * (c.Fe3 + c.FeS + c.FeS2)
    m_fe = np.sum(dz * fe_total_bulk[:-1]).value
    time_str = f" Time: {get_time_units(total_time):.2f~P}"
    
    if mp.isotopes:
        d34s = get_total_delta(c, mp)
        fes_mask = c.FeS.value > 1e-3
        d_fes = get_delta(c.FeS.value[fes_mask], c.FeS_32.value[fes_mask], mp.VCDT) if np.any(fes_mask) else np.array([])
        v_fes = d_fes[~np.isnan(d_fes)]
        min_dFeS = float(np.min(v_fes)) if len(v_fes) > 0 else np.nan
        max_dFeS = float(np.max(v_fes)) if len(v_fes) > 0 else np.nan

        ts2_mask = c.TS2.value > 1e-3
        d_ts2 = get_delta(c.TS2.value[ts2_mask], c.TS2_32.value[ts2_mask], mp.VCDT) if np.any(ts2_mask) else np.array([])
        v_ts2 = d_ts2[~np.isnan(d_ts2)]
        min_dTS2 = float(np.min(v_ts2)) if len(v_ts2) > 0 else np.nan
        max_dTS2 = float(np.max(v_ts2)) if len(v_ts2) > 0 else np.nan

        _log(
            f"Step {step:4d}, {time_str}, "
            f"dt: {get_time_units(current_dt):.2f~P}, RMS: {rms_change:.2e}, "
            f"d34S = {d34s:.2f}, "
            f"dTS2: [{min_dTS2:.1f}, {max_dTS2:.1f}]‰, "
            f"dFeS: [{min_dFeS:.1f}, {max_dFeS:.1f}]‰"
        )
        if mp.title is None:
            title_str = time_str + r", $\delta^{34}$S = " + f"{d34s:.1f} [mUr]"
        else:
            title_str = mp.title
    else:
        _log(
            f"Step {step:4d}, {time_str}, "
            f"dt: {get_time_units(current_dt):.2f~P}, RMS Chg: {rms_change:.2e}, "
            f"Total Fe {m_fe:.2e}"
        )
        if mp.title is None:
            title_str = time_str
        else:
            title_str = mp.title

    if plot_queue is not None:
        write_to_queue_async(
            plot_queue,
            mp,
            c,
            mp.k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
            current_dt,
            title_str,
        )
    else:
        save_data_async(
            mp,
            c,
            mp.k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
            current_dt,
            title=title_str,
        )


def run_non_steady_state_solver_coupled(
    mp: Any,
    c: Any,
    species_list_full: List[str],
    species_list_partial: List[str],
    k: Any,
    diagenetic_reactions: Any,
    equilibrium_reactions: Any,
    mesh: Mesh,
    D_mol: Any,
    bc_map: Dict[str, Any],
    z: np.ndarray,
    plot_queue: Optional[Any] = None,
) -> Tuple[int, float]:
    """
    Solves the non-steady state ADR coupled system with advanced adaptive time stepping.

    This function splits the model into manageable steps:
    1. Pre-builds passive transport terms.
    2. Runs a time loop where reaction terms are updated and solved.
    3. Adapts the time step using a PID-controlled logic.
    """
    from .diff_lib import get_delta, get_total_delta, save_state

    start_wall = time.time()
    solver = _get_solver(mp)

    log_path = f"{mp.plot_name}.log"
    _log_file = open(log_path, "w", buffering=1)

    def _log(msg: str) -> None:
        print(msg, flush=True)
        _log_file.write(msg + "\n")

    # --- Initialize Adaptive Time Stepping ---
    dt_controller = AdaptiveDT(
        dt_min=mp.dt_min,
        dt_max=mp.dt_max,
        dt_initial=getattr(mp, "dt_init", mp.dt_min),
    )

    # --- Initialize Rate-Change-Based Timestep Adaptation ---
    enable_rate_adaptation = getattr(mp, "enable_rate_adaptation", False)
    monitored_rate_species = getattr(mp, "monitored_rate_species", ["FeS", "TS2"])
    rate_threshold = getattr(mp, "rate_threshold", 1e-8)
    rate_adaptation_start_step = getattr(mp, "rate_adaptation_start_step", 3)
    enable_rate_magnitude_check = getattr(mp, "enable_rate_magnitude_check", False)
    prev_rates = {}
    prev_rates_2 = {}
    prev_dt = getattr(mp, "dt_init", mp.dt_min)
    prev_dt_2 = prev_dt

    # --- Initialize Dynamic Isotope dt Limiter ---
    enable_isotope_dt_limiter = getattr(mp, "enable_isotope_dt_limiter", False)
    isotope_limiter_species = getattr(mp, "isotope_limiter_species", "FeS")
    isotope_onset_threshold = getattr(mp, "isotope_onset_threshold", 1e-5)

    if enable_isotope_dt_limiter and getattr(mp, "isotopes", False):
        dt_max_isotope = _calculate_dt_max_isotope(mp, k, D_mol)
        mp.dt_max_isotope = dt_max_isotope
        w = max(getattr(mp, "w", 0.0), 1e-20)
        dz = getattr(mp, "reaction_zone_spacing", 0.0001)
        msg = f"Isotope dt Limiter: Calculated dt_max_isotope = {get_time_units(dt_max_isotope):.4f~P} (dz = {dz*1000:.3f} mm, w = {w*3.1536e7*100:.3f} cm/yr)"
        _log(msg)
        print(msg)

    # Global CFL bound (optional safety)
    dx = mesh.cellVolumes.min() ** (1 / mesh.dim)
    v_max = max(abs(getattr(mp, "w", 0.0)), abs(getattr(mp, "advection", 0.0)))
    D_max = 0.0
    for s in species_list_partial:
        D_s = np.max(getattr(D_mol, s) + D_mol.D_bio)
        D_max = max(D_max, D_s)

    dt_cfl = dt_controller.cfl_limit(dx, v_max, D_max)

    print(
        f"Starting Adaptive ADR Solver. dt_init: {get_time_units(dt_controller.dt):.2f~P}"
    )

    # Build the transport backbone
    species_struct, passive_eqs = _build_passive_eqs(
        mp, c, mesh, D_mol, bc_map, species_list_partial
    )

    # Setup static coupled equation and coefficient variables (Option C)
    coupled_eq, LHS_vars, RHS_vars, CROSS_vars = _setup_static_coupled_equation(
        mp,
        c,
        k,
        mesh,
        passive_eqs,
        species_struct,
        diagenetic_reactions,
        species_list_partial,
    )

    step = 0
    total_time = mp.start_time
    status = "Maximum steps or simulation time reached"
    max_change = 0.0
    title_str = ""

    try:
        while total_time < mp.t_end and step < mp.max_steps:
            step += 1
            current_dt = dt_controller.dt

            # updateOld() stores current -> var.old; used below for RMS and restore
            for s_obj in species_struct:
                s_obj["var"].updateOld()

            # --- Solve Step (with automatic retry on failure) ---
            converged = False
            RATES_tentative = {}
            while not converged:
                mp.current_dt = current_dt
                try:
                    # Update static coefficient variables in-place
                    RATES = _update_static_coefficients(
                        mp,
                        c,
                        k,
                        diagenetic_reactions,
                        LHS_vars,
                        RHS_vars,
                        CROSS_vars,
                        species_list_partial,
                    )

                    coupled_eq.sweep(
                        dt=current_dt,
                        solver=solver,
                    )
                    converged = True

                    # post transport clips
                    mp.in_clip = True
                    try:
                        equilibrium_reactions(mp, c, k, None, RATES, current_dt)
                    finally:
                        mp.in_clip = False

                    if enable_rate_adaptation:
                        RATES_tentative = _update_static_coefficients(
                            mp,
                            c,
                            k,
                            diagenetic_reactions,
                            LHS_vars,
                            RHS_vars,
                            CROSS_vars,
                            species_list_partial,
                        )

                except Exception as e:
                    tb_str = "".join(
                        traceback.format_exception(type(e), e, e.__traceback__)
                    )
                    print(
                        f"  Step failed at dt={get_time_units(current_dt):.2f~P}:\n\n {tb_str}\n Cutting dt."
                    )
                    # Restore state from FiPy's built-in old-value store
                    for s_obj in species_struct:
                        s_obj["var"].value[:] = s_obj["var"].old.value

                    # Cut time step and retry
                    current_dt = dt_controller.update(0.0, step_success=False)
                    if current_dt <= mp.dt_min * 1.01:
                        raise RuntimeError(
                            "Solver failed and time step is already at minimum."
                        )

                # --- Rate Validation Check (A Posteriori) ---
                if converged and enable_rate_adaptation and step >= rate_adaptation_start_step and prev_rates:
                    violation, violation_reason = _validate_rates(
                        monitored_rate_species,
                        RATES_tentative,
                        prev_rates,
                        prev_rates_2,
                        current_dt,
                        prev_dt,
                        prev_dt_2,
                        rate_threshold,
                        enable_rate_magnitude_check,
                    )
                    if violation:
                        _log(f"  Step rejected at dt={get_time_units(current_dt):.4f~P}: {violation_reason}. Rollback.")
                        for s_obj in species_struct:
                            s_obj["var"].value[:] = s_obj["var"].old.value
                        
                        converged = False
                        current_dt = dt_controller.update(0.0, step_success=False)
                        if current_dt <= mp.dt_min * 1.01:
                            raise RuntimeError(
                                "Rate validation failed and time step is already at minimum."
                            )

            # --- Calculate Convergence Metrics ---
            rms_change = max(
                float(
                    np.sqrt(np.mean((s_obj["var"].value - s_obj["var"].old.value) ** 2))
                )
                for s_obj in species_struct
            )

            # --- Dynamically adapt solver tolerance to prevent false steady-state convergence ---
            if getattr(mp, "adaptive_solver_tolerance", False):
                new_tol = max(mp.tolerance, min(1e-4, rms_change * 0.1))
                solver.tolerance = new_tol

            total_time += current_dt

            # --- Update Rate History for Next Step ---
            if enable_rate_adaptation:
                prev_dt_2 = prev_dt
                prev_dt = current_dt
                for name in monitored_rate_species:
                    if name in RATES_tentative:
                        if name in prev_rates:
                            prev_rates_2[name] = np.copy(prev_rates[name])
                        prev_rates[name] = np.copy(RATES_tentative[name])

            # --- Adapt Time Step for Next Iteration ---
            if enable_rate_adaptation:
                dt_controller._dt = min(dt_controller._dt * dt_controller.growth_factor, dt_controller.dt_max)
                dt_controller._dt_prev = dt_controller._dt
            else:
                adaptive_target = getattr(mp, "dt_target_change", 1e-4)
                dt_controller.update(
                    error_metric=rms_change,
                    dt_cfl=None,
                    step_success=True,
                    target_error=adaptive_target,
                )

            # --- Apply Dynamic Isotope dt Limiter ---
            if enable_isotope_dt_limiter and getattr(mp, "isotopes", False):
                if isotope_limiter_species in c:
                    max_conc = np.max(c[isotope_limiter_species].value)
                    if max_conc > isotope_onset_threshold:
                        dt_controller._dt = min(dt_controller._dt, dt_max_isotope)
                        dt_controller._dt_prev = min(dt_controller._dt_prev, dt_max_isotope)

            if step % mp.backup_step == 0:
                gc.collect()
                save_state(c, f"{mp.plot_name}_bak.npz")

            # Reporting
            if step % mp.report_step == 0:
                _report_step_status(
                    step,
                    total_time,
                    current_dt,
                    rms_change,
                    mp,
                    c,
                    z,
                    species_list_full,
                    D_mol,
                    diagenetic_reactions,
                    equilibrium_reactions,
                    plot_queue,
                    _log,
                )

            # Steady State Check
            if rms_change < mp.dt_tolerance:
                _log(
                    f"Steady State Met: rms_change {rms_change:.2e} < tolerance {mp.dt_tolerance:.2e}"
                )
                status = "Steady State Converged"
                dt = get_time_units(mp.dt_max)
                if hasattr(c, "Fe3") and hasattr(c, "Fe2_total"):
                    Fe3_lost = c.Fe3.value[0] - c.Fe3.value[-1]
                    Fe2_gained = c.Fe2_total.value[-1] - c.Fe2_total.value[0]
                    _log(
                        f"dt = {dt:~P.2f}, Fe3_lost = {Fe3_lost:.2f}, Fe2_gained = {Fe2_gained:.2f}"
                    )
                else:
                    _log(f"dt = {dt:~P.2f}")
                break

    except KeyboardInterrupt:
        status = "Solver interrupted by user"
    except Exception as e:
        status = f"Solver crashed: {e}"
        print(traceback.format_exc())

    # Final Save
    _log(
        f"Final Report: {status} in {step} steps. Total Wall Time: {time.time() - start_wall:.2f}s"
    )
    _log_file.close()

    # Always write the final data and state synchronously to prevent data loss on termination/interrupt
    try:
        csv_file = f"{mp.plot_name}.csv"
        state_file = f"{mp.plot_name}_state.npz"
        print(
            f"Saving final results to {csv_file} and state to {state_file} ...",
            flush=True,
        )
        save_data(
            mp,
            c,
            k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
        )
        save_state(c, state_file)
    except Exception as e:
        print(f"Error during final synchronous save: {e}", flush=True)

    if plot_queue is not None:
        write_to_queue_async(
            plot_queue,
            mp,
            c,
            k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
            current_dt,
            title_str,
        )

    return step, rms_change
