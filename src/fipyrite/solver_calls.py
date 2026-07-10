"""Build the equation matrix and call the respective solvers."""

from __future__ import annotations

import gc
import math
import time
import traceback
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
        PETSc.Options().setValue("ksp_converged_use_initial_residual_norm", "")

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
        vel = mp.w - mp.advection if props["type"] == "dissolved" else mp.w
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
    #     import numpy as np
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

    # Global CFL bound (optional safety)
    dx = mesh.cellVolumes.min() ** (1 / mesh.dim)
    v_max = max(abs(mp.w), abs(mp.advection))
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
                        # underRelaxation=0.9, # currently failing
                    )
                    converged = True

                    # post transport clips
                    mp.in_clip = True
                    try:
                        equilibrium_reactions(mp, c, k, None, RATES, current_dt)
                    finally:
                        mp.in_clip = False

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

            # --- Calculate Convergence Metrics ---
            # Using Normalized RMS change to reduce sensitivity to grid resolution
            # RMS = sqrt( mean( (c_new - c_old)**2 ) )
            rms_change = max(
                float(
                    np.sqrt(np.mean((s_obj["var"].value - s_obj["var"].old.value) ** 2))
                )
                for s_obj in species_struct
            )

            # --- Dynamically adapt solver tolerance to prevent false steady-state convergence ---
            if getattr(mp, "adaptive_solver_tolerance", False):
                # Keep solver tolerance tighter than current step-to-step changes (rms_change),
                # capped between mp.tolerance (minimum) and 1e-4 (maximum/loosest).
                new_tol = max(mp.tolerance, min(1e-4, rms_change * 0.1))
                solver.tolerance = new_tol

            total_time += current_dt

            # --- Adapt Time Step for Next Iteration ---
            # Use dt_target_change for adaptation, but default to something sane if missing
            adaptive_target = getattr(mp, "dt_target_change", 1e-4)

            dt_controller.update(
                error_metric=rms_change,
                dt_cfl=None,  # DISABLE hard CFL cap for implicit solver
                step_success=True,
                target_error=adaptive_target,
            )
            if step % mp.backup_step == 0:
                gc.collect()
                save_state(c, f"{mp.plot_name}_bak.npz")

            # Reporting
            if step % mp.report_step == 0:  # Force reporting for debug
                phi = mp.phi
                dz = np.diff(z)
                fe_total_bulk = phi * c.Fe2_total + (1 - phi) * (c.Fe3 + c.FeS + c.FeS2)
                m_fe = np.sum(dz * fe_total_bulk[:-1]).value
                time_str = f" Time: {get_time_units(total_time):.2f~P}"
                if mp.isotopes:
                    d34s = get_total_delta(c, mp)
                    _log(
                        f"Step {step:4d}, {time_str}, "
                        f"dt: {get_time_units(current_dt):.2f~P}, RMS Chg: {rms_change:.2e}, "
                        f"d34S = {d34s:.2f}, "
                        f"Total Fe {m_fe:.2e}"
                    )
                    d34S = get_delta(2 * c.FeS2[-1], c.FeS2_32[-1], mp.VCDT)
                    if mp.title is None:
                        title_str = (
                            time_str + r", $\delta^{34}$S = " + f"{d34s:.1f} [mUr]"
                        )
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
                        k,
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
                        k,
                        species_list_full,
                        z,
                        D_mol,
                        diagenetic_reactions,
                        equilibrium_reactions,
                        current_dt,
                        title=title_str,
                    )

            # Steady State Check
            # if c.Fe3[-10] < 70:
            if rms_change < mp.dt_tolerance:
                _log(
                    f"Steady State Met: rms_change {rms_change:.2e} < tolerance {mp.dt_tolerance:.2e}"
                )
                status = "Steady State Converged"
                Fe3_lost = c.Fe3.value[0] - c.Fe3.value[-1]
                Fe2_gained = c.Fe2_total.value[-1] - c.Fe2_total.value[0]
                dt = get_time_units(mp.dt_max)
                _log(
                    f"dt = {dt:~P.2f}, Fe3_lost = {Fe3_lost:.2f}, Fe2_gained = {Fe2_gained:.2f}"
                )
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
