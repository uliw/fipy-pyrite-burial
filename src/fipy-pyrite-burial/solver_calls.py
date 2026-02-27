from __future__ import annotations

import time
import traceback
from functools import reduce
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from fipy import CellVariable, LinearLUSolver
from fipy.terms.diffusionTerm import DiffusionTerm
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
from fipy.terms.transientTerm import TransientTerm

# Functions that live elsewhere in the project – we only need the signatures.
# They are imported lazily inside the helpers that actually use them.
#   get_time_units, save_data_async, data_container, equilibrium_reactions
#   (and the user‑provided `diagenetic_reactions`) are assumed to be available.


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def select_solver(mp: Any) -> Any:
    """Return a FiPy solver instance according to ``mp.solver_backend``."""
    if mp.solver_backend == "LinearGMRESSolver":
        from fipy.solvers.petsc import LinearGMRESSolver as PETScGMRESSolver

        return PETScGMRESSolver(precon="hypre", tolerance=mp.tolerance)
    if mp.solver_backend == "PETScLUSolver":
        from fipy.solvers.petsc import PETScLUSolver

        return PETScLUSolver(tolerance=mp.tolerance)
    if mp.solver_backend == "LinearLUSolver":
        return LinearLUSolver(tolerance=mp.tolerance)
    if mp.solver_backend == "PETScNewtonSolver":
        from fipy.solvers.petsc import PETScNewtonSolver

        return PETScNewtonSolver(
            precon="hypre", tolerance=mp.tolerance, max_it=30, damping=0.5
        )
    # Fallback – FiPy’s default linear solver
    from fipy import DefaultSolver

    return DefaultSolver(tolerance=mp.tolerance)


def build_variables(
    mesh: Any,
    species_list_partial: List[str],
    diagenetic_reactions: Callable,
    mp: Any,
    c: Any,
    D_mol: Any,
    k: data_container,
) -> Tuple[
    Dict[str, CellVariable],
    Dict[str, CellVariable],
    Dict[str, Dict[str, CellVariable]],
]:
    """
    Create the LHS, RHS and cross‑species FiPy variables for the *partial* species.

    Returns
    -------
    lhs_vars, rhs_vars, cross_vars
    """
    from diff_lib import data_container

    lhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}
    rhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}
    cross_vars = {s: {} for s in species_list_partial}

    # Populate ``cross_vars`` – we need a dummy call to obtain the topology.
    f_init, _ = diagenetic_reactions(mp, c, k, f=data_container())
    for s in species_list_partial:
        res = getattr(f_init, s)
        if len(res) > 3:
            for source_name, _ in res[3]:
                if source_name not in cross_vars[s]:
                    cross_vars[s][source_name] = CellVariable(mesh=mesh, value=0.0)

    return lhs_vars, rhs_vars, cross_vars


def attach_source_variables(
    variables: Dict[str, fp.CellVariable],
    LHS: Dict[str, Union[float, List[fp.ImplicitSourceTerm]]],
    CROSS: Dict[str, List[Tuple[str, float]]],
) -> None:
    """
    Walk through LHS and CROSS and set ``term.var`` to the actual
    ``CellVariable`` objects.  Called once per time step, just before the
    global FiPy equations are assembled.
    """
    # First, fix the terms that were already stored in LHS (precipitation, etc.)
    for sp, term_list in LHS.items():
        if isinstance(term_list, list):
            for term in term_list:
                if term.var is None:  # only the placeholders
                    term.var = variables[sp]

    # Then handle the coupling entries that are still in CROSS.
    for target, pairs in CROSS.items():
        target_var = variables[target]
        for source_name, coeff in pairs:
            source_var = variables[source_name]
            term = fp.ImplicitSourceTerm(var=target_var, coeff=coeff * source_var)
            LHS.setdefault(target, []).append(term)


def build_equations(
    mesh: Any,
    species_list_partial: List[str],
    bc_map: Dict[str, Dict[str, Any]],
    c: Any,
    D_mol: Any,
    mp: Any,
    lhs_vars: Dict[str, CellVariable],
    rhs_vars: Dict[str, CellVariable],
    cross_vars: Dict[str, Dict[str, CellVariable]],
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Assemble transport + reaction terms for each partial species and return the
    coupled FiPy equation together with a cached description of each species.
    """
    species_struct: List[Dict[str, Any]] = []
    eqs: List[Any] = []

    for name in species_list_partial:
        var = getattr(c, name)
        props = bc_map[name]

        species_struct.append(
            {
                "name": name,
                "var": var,
                "lhs": lhs_vars[name],
                "rhs": rhs_vars[name],
                "cross": cross_vars[name],
            }
        )

        # ---- Transport -------------------------------------------------------
        D_total = np.maximum(getattr(D_mol, name) + D_mol.D_bio, 1e-20)
        vel = mp.w - mp.advection if props["type"] == "dissolved" else mp.w
        u_var = CellVariable(mesh=mesh, value=vel, rank=1)

        conv_term = PowerLawConvectionTerm(coeff=u_var, var=var)
        diff_term = DiffusionTerm(coeff=CellVariable(mesh=mesh, value=D_total), var=var)

        # ---- Reaction --------------------------------------------------------
        lhs_term = ImplicitSourceTerm(coeff=lhs_vars[name], var=var)
        rhs_term = rhs_vars[name]

        cross_term = 0.0
        for src_name, cv in cross_vars[name].items():
            src_var = getattr(c, src_name)
            cross_term += ImplicitSourceTerm(coeff=cv, var=src_var)

        # ---- Irrigation (only for dissolved) ---------------------------------
        irr_term = 0.0
        if props["type"] == "dissolved":
            irr_term = ImplicitSourceTerm(
                coeff=CellVariable(mesh=mesh, value=-D_mol.D_irr), var=var
            ) + CellVariable(mesh=mesh, value=D_mol.D_irr * props["top"])

        # ---- Full equation ----------------------------------------------------
        eq = (TransientTerm(var=var) + conv_term) == (
            diff_term + lhs_term + rhs_term + cross_term + irr_term
        )
        eqs.append(eq)

    coupled_eq = reduce(lambda a, b: a & b, eqs)
    return coupled_eq, species_struct


def update_reaction_rates(
    mp: Any,
    c: Any,
    k: Any,
    diagenetic_reactions: Callable,
    species_struct: List[Dict[str, Any]],
    species_list_partial: List[str],
    current_dt: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute equilibrium and diagenetic reaction rates, merge them and write the
    new matrix coefficients into the cached FiPy variables.
    """
    from diff_lib import data_container
    from reactions_new import (
        equilibrium_reactions,
    )  # local import to avoid circular deps

    # ---- Equilibrium (operator‑splitting) ------------------------------------
    _, rates_eq = equilibrium_reactions(mp, c, k, None, {}, current_dt)

    # ---- Diagenetic reactions ------------------------------------------------
    f_res, rates = diagenetic_reactions(mp, c, k, f=data_container())

    # Merge equilibrium rates into diagenetic ones (for reporting only)
    for sp in rates:
        if sp in rates_eq:
            rates[sp] += rates_eq[sp]

    # ---- Update LHS / RHS / cross coefficients -------------------------------
    for struct in species_struct:
        name = struct["name"]
        res_tuple = getattr(f_res, name)

        # Diagonal (LHS) and source (RHS)
        struct["lhs"].setValue(getattr(res_tuple[0], "value", res_tuple[0]))
        struct["rhs"].setValue(getattr(res_tuple[1], "value", res_tuple[1]))

        # Cross‑terms, if any
        if len(res_tuple) > 3:
            batch = {src: 0.0 for src in struct["cross"]}
            for src_name, coeff in res_tuple[3]:
                if src_name in batch:
                    batch[src_name] += getattr(coeff, "value", coeff)

            for src, val in batch.items():
                struct["cross"][src].setValue(val)

    return f_res, rates


def adaptive_step(
    coupled_eq: Any,
    solver: Any,
    current_dt: float,
    species_struct: List[Dict[str, Any]],
    mp: Any,
    c: Any,
    k: Any,
    diagenetic_reactions: Callable,
    species_list_partial: List[str],
) -> Tuple[bool, float, Dict[str, np.ndarray]]:
    """
    Perform one pseudo‑transient step.  If the linear solve fails, the time step
    is reduced and the step is retried.  The function returns a tuple::

        (step_converged, new_dt, backup_values)

    where ``backup_values`` is a mapping ``species_name → old_value`` used by
    the caller for convergence diagnostics.
    """
    # Save the current solution (so we can roll back on failure)
    backup = {s["name"]: s["var"].value.copy() for s in species_struct}
    cut_factor = 0.5

    while True:
        try:
            # Update reaction rates *before* solving
            update_reaction_rates(
                mp,
                c,
                k,
                diagenetic_reactions,
                species_struct,
                species_list_partial,
                current_dt,
            )

            # Attach the real FiPy variables to every placeholder term
            variables = {s["name"]: s["var"] for s in species_struct}
            attach_source_variables(variables, lhs_vars, cross_vars)

            # Sweep the coupled system
            coupled_eq.sweep(dt=current_dt, solver=solver)
            return True, current_dt, backup
        except Exception as exc:  # noqa: BLE001 (we want to catch FiPy failures)
            print(
                f"  Step failed at dt={current_dt:.1e}: {exc}. "
                f"Retrying with dt*{cut_factor:.2f}"
            )
            traceback.print_exc()
            current_dt *= cut_factor
            if current_dt < 1e-5:
                raise RuntimeError(
                    "Time step became too small – model appears stiff."
                ) from exc

            # Restore previous values before retrying
            for s in species_struct:
                s["var"].value[:] = backup[s["name"]]


def report_progress(
    step: int,
    mp: Any,
    total_time: float,
    current_dt: float,
    max_change: float,
    c: Any,
    k: Any,
    species_list_full: List[str],
    z: Any,
    D_mol: Any,
    diagenetic_reactions: Callable,
) -> None:
    """Print a status line and trigger asynchronous saving if required."""
    if step % mp.report_step == 0:
        print(
            f"Time: {get_time_units(total_time):.2f~P}, "
            f"dt: {get_time_units(current_dt):.2f~P}, "
            f"Max Change: {max_change:.2e}"
        )
        # The original code saved the data asynchronously; we keep that call.
        save_data_async(
            mp, c, k, species_list_full, z, D_mol, diagenetic_reactions, current_dt
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public function – thin wrapper that orchestrates the helpers
# ──────────────────────────────────────────────────────────────────────────────
def run_non_steady_state_solver_coupled(
    mp: Any,
    c: Any,
    species_list_full: List[str],
    species_list_partial: List[str],
    k: Any,
    diagenetic_reactions: Callable,
    mesh: Any,
    D_mol: Any,
    bc_map: Dict[str, Dict[str, Any]],
    z: Any,
) -> Tuple[int, float]:
    """
    Solve the coupled transport‑reaction system using an adaptive pseudo‑transient
    scheme.

    Parameters
    ----------
    mp
        Model‑parameter container (holds ``solver_backend``, tolerances, etc.).
    c
        Container with FiPy ``CellVariable`` fields for each species.
    species_list_full
        List of *all* species (used only for final data output).
    species_list_partial
        Sub‑set of species that appear in the transport‑reaction system.
    k
        Kinetic‑parameter container passed to the reaction functions.
    diagenetic_reactions
        Callable ``(mp, c, k, f) → (f_res, rates)``.
    mesh
        FiPy mesh on which the variables live.
    D_mol
        Diffusivity container (has attributes for each species plus ``D_bio`` &
        ``D_irr``).
    bc_map
        Boundary‑condition map keyed by species name.
    z
        Depth coordinate array (only used for output).

    Returns
    -------
    steps_done : int
        Number of pseudo‑transient steps performed.
    final_max_change : float
        Largest absolute change in any variable during the last successful step.
    """
    from diff_lib import get_time_units, save_data_async

    start_wall = time.time()
    # --------------------------------------------------------------------- #
    # 1️⃣ Solver selection
    # --------------------------------------------------------------------- #
    solver = select_solver(mp)

    # --------------------------------------------------------------------- #
    # 2️⃣ Build FiPy variables (LHS/RHS/cross) and equations
    # --------------------------------------------------------------------- #
    lhs_vars, rhs_vars, cross_vars = build_variables(
        mesh, species_list_partial, diagenetic_reactions, mp, c, D_mol, k
    )
    coupled_eq, species_struct = build_equations(
        mesh,
        species_list_partial,
        bc_map,
        c,
        D_mol,
        mp,
        lhs_vars,
        rhs_vars,
        cross_vars,
    )

    # --------------------------------------------------------------------- #
    # 3️⃣ Adaptive time‑stepping loop
    # --------------------------------------------------------------------- #
    dt_min = getattr(mp, "dt_min", 1e-5 * 365 * 24 * 3600)  # default: ≈ minutes
    current_dt = dt_min
    dt_max = mp.dt_max
    growth_factor = 1.2
    cut_factor = 0.5

    print(
        f"Starting Pseudo‑Transient Solver (Adaptive dt) "
        f"{get_time_units(current_dt):.2f~P}"
    )

    step = 0
    total_time = 0.0
    last_max_change = np.inf
    status = "max steps or time reached"

    try:
        while total_time < mp.t_end and step < mp.max_steps:
            step += 1

            # Advance FiPy variables to the new time level
            for s in species_struct:
                s["var"].updateOld()

            # Keep a copy of the solution before we attempt the step
            backup = {s["name"]: s["var"].value.copy() for s in species_struct}

            # -------------------------------------------------------------- #
            # Adaptive step – may shrink dt and retry internally
            # -------------------------------------------------------------- #
            step_ok, current_dt, _ = adaptive_step(
                coupled_eq,
                solver,
                current_dt,
                species_struct,
                mp,
                c,
                k,
                diagenetic_reactions,
                species_list_partial,
            )
            if not step_ok:
                raise RuntimeError("Adaptive step could not converge.")  # safety

            # -------------------------------------------------------------- #
            # Compute max change (steady‑state criterion)
            # -------------------------------------------------------------- #
            max_change = max(
                np.max(np.abs(s["var"].value - backup[s["name"]]))
                for s in species_struct
            )
            total_time += current_dt

            # -------------------------------------------------------------- #
            # Adapt the time step based on convergence trend
            # -------------------------------------------------------------- #
            if step > 1:
                if max_change < last_max_change:
                    current_dt = min(current_dt * growth_factor, dt_max)
                else:
                    current_dt = max(current_dt * cut_factor, dt_min)
            else:
                # First step – use the original tolerance logic
                if max_change < mp.dt_tolerance:
                    current_dt = min(current_dt * growth_factor, dt_max)

            last_max_change = max_change

            # -------------------------------------------------------------- #
            # Reporting / data output
            # -------------------------------------------------------------- #
            report_progress(
                step,
                mp,
                total_time,
                current_dt,
                max_change,
                c,
                k,
                species_list_full,
                z,
                D_mol,
                diagenetic_reactions,
            )

            # -------------------------------------------------------------- #
            # Steady‑state check
            # -------------------------------------------------------------- #
            if max_change < mp.dt_tolerance:
                print(f"Steady State criteria met (max_change = {max_change:.2e})")
                status = "Converged" if step < mp.max_steps else "Failed"
                break

    except KeyboardInterrupt:
        status = "Interrupted"
    finally:
        # ----------------------------------------------------------------- #
        # Final data write and wall‑time report
        # ----------------------------------------------------------------- #
        save_data_async(
            mp, c, k, species_list_full, z, D_mol, diagenetic_reactions, current_dt
        )
        wall = time.time() - start_wall
        print(f"{status} in {step} steps. Wall time: {wall:.2f}s")

    return step, max_change
