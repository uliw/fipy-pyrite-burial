"""Build the equation matrix.

and call the respective solvers.
"""

from diff_lib import get_time_units, data_container, save_data_async


def run_non_steady_state_solver_coupled(
    mp,
    c,
    species_list_full,
    species_list_partial,
    k,
    diagenetic_reactions,
    mesh,
    D_mol,
    bc_map,
    z,
):
    import time
    import numpy as np

    # from fipy import *
    from fipy import CellVariable

    # from fipy import LinearLUSolver
    from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
    from fipy.terms.diffusionTerm import DiffusionTerm
    from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
    from fipy.terms.transientTerm import TransientTerm
    from functools import reduce
    from reactions_new import equilibrium_reactions
    import traceback

    # from fipy.solvers.petsc import LinearLUSolver as PETScLUSolver
    from fipy.solvers.petsc import LinearGMRESSolver as PETScLUSolver

    # solver = PETScLUSolver(tolerance=mp.tolerance, iterations=1)
    solver = PETScLUSolver(precon="hypre", tolerance=mp.tolerance)

    start_wall = time.time()

    # --- OPTIMIZATION 1: Adaptive Time Stepping Setup ---
    current_dt = getattr(
        mp, "dt_min", 1e-5 * 365 * 24 * 3600
    )  # Start small (e.g., minutes/hours)
    dt_max = mp.dt_max  # Cap at your large step
    growth_factor = 1.2  # Grow by 20% on success
    cut_factor = 0.5  # Cut in half on failure

    print(
        f"Starting Pseudo-Transient Solver (Adaptive dt) {get_time_units(current_dt):.2f~P}"
    )

    # 1. PRE-BUILD THE EQUATIONS (Your code was good here)
    # --------------------------
    lhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}
    rhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}
    cross_vars = {s: {} for s in species_list_partial}

    # ... [Your Topology Analysis Code is good, keeping it as is] ...
    # (Assume cross_vars is populated here)
    f_init, RATES = diagenetic_reactions(
        mp, c, k, f=data_container()
    )  # dummy call for topology
    for s in species_list_partial:
        res = getattr(f_init, s)
        if len(res) > 3:
            for source_name, _ in res[3]:
                if source_name not in cross_vars[s]:
                    cross_vars[s][source_name] = CellVariable(mesh=mesh, value=0.0)

    eqs = []

    # --- OPTIMIZATION 2: Cache Variables to avoid getattr in loop ---
    # We store tuples of (Variable, LHS_Coeff_Var, RHS_Var, Cross_Dict)
    species_struct = []

    for species_name in species_list_partial:
        var = getattr(c, species_name)
        props = bc_map[species_name]

        # Cache these objects for the loop
        species_struct.append(
            {
                "name": species_name,
                "var": var,
                "lhs": lhs_vars[species_name],
                "rhs": rhs_vars[species_name],
                "cross": cross_vars[species_name],
            }
        )

        # -- Transport Construction (Same as your code) --
        D_total = np.maximum(getattr(D_mol, species_name) + D_mol.D_bio, 1e-20)
        vel = mp.w - mp.advection if props["type"] == "dissolved" else mp.w

        u_var = CellVariable(mesh=mesh, value=([vel],), rank=1)
        conv_term = PowerLawConvectionTerm(coeff=u_var, var=var)
        diff_term = DiffusionTerm(coeff=CellVariable(mesh=mesh, value=D_total), var=var)

        # -- Reaction Terms --
        lhs_term = ImplicitSourceTerm(coeff=lhs_vars[species_name], var=var)
        rhs_term = rhs_vars[species_name]

        cross_terms = 0.0
        for src_name, cv in cross_vars[species_name].items():
            src_var = getattr(c, src_name)
            cross_terms += ImplicitSourceTerm(coeff=cv, var=src_var)

        # -- Irrigation --
        irr_term = 0.0
        if props["type"] == "dissolved":
            irr_term = ImplicitSourceTerm(
                coeff=CellVariable(mesh=mesh, value=-D_mol.D_irr), var=var
            ) + CellVariable(mesh=mesh, value=D_mol.D_irr * props["top"])

        eq = (
            TransientTerm(var=var) + conv_term
            == diff_term + lhs_term + rhs_term + cross_terms + irr_term
        )
        eqs.append(eq)

    # Couple them
    coupled_eq = reduce(lambda a, b: a & b, eqs)

    # 2. TIME STEPPING LOOP
    step = 0
    total_time = 0.0
    last_max_change = 1e10  # Initial large value

    while total_time < mp.t_end and step < mp.max_steps:  # Or check convergence
        step += 1

        # A. Update Old Values (Advance Time)
        for s_obj in species_struct:
            s_obj["var"].updateOld()

        # Snapshot for checking if this step succeeds
        last_sol_backup = {s["name"]: s["var"].value.copy() for s in species_struct}

        # --- Adaptive Loop: Retry step if solver fails ---
        step_converged = False
        while not step_converged:
            try:
                # B. Update Reaction Rates
                # Initialize equilibrium rates container
                RATES_eq = {s: np.zeros_like(c.so4.value) for s in species_list_partial}

                # Apply Instantaneous Equilibrium (Operator Splitting) - NOW BEFORE SOLVER
                # This modifies c.fe2, c.h2s, c.fes IN PLACE
                # We pass None for 'f' as it's not used in equilibrium_reactions
                _, RATES_eq = equilibrium_reactions(
                    mp, c, k, None, RATES_eq, current_dt
                )

                # Note: We pass the CURRENT guess 'c' to reaction function
                f_res, RATES = diagenetic_reactions(mp, c, k, f=data_container())

                # Merge RATES_eq into RATES for reporting
                for s in RATES:
                    if s in RATES_eq:
                        RATES[s] += RATES_eq[s]

                # Update Matrix Coefficients (Fast, using cached objects)
                for s_obj in species_struct:
                    name = s_obj["name"]
                    res_tuple = getattr(f_res, name)

                    # Update Diagonal
                    s_obj["lhs"].setValue(getattr(res_tuple[0], "value", res_tuple[0]))
                    s_obj["rhs"].setValue(getattr(res_tuple[1], "value", res_tuple[1]))

                    # Update Cross-Terms
                    if len(res_tuple) > 3:
                        # Reset batch accumulator
                        batch_coeffs = {src: 0.0 for src in s_obj["cross"]}
                        for source_name, coeff in res_tuple[3]:
                            if source_name in batch_coeffs:
                                val = getattr(coeff, "value", coeff)
                                batch_coeffs[source_name] += val

                        # Push to FiPy variables
                        for src, val in batch_coeffs.items():
                            s_obj["cross"][src].setValue(val)

                # C. Sweep the Coupled System
                # Note: 'dt' is now dynamic
                res = coupled_eq.sweep(dt=current_dt, solver=solver)

                # D. Apply Instantaneous Equilibrium (Operator Splitting) -- MOVED ABOVE
                # This matches user request to have equilibrium proceed transport/matrix solve

                # If we get here without error, the linear solve worked.
                step_converged = True

            except Exception as e:
                # If solver diverged or crashed
                print(
                    f"  Step failed at dt={current_dt:.1e}: {e}. Retrying with smaller dt."
                )
                traceback.print_exc()
                current_dt *= cut_factor
                if current_dt < 1e-5:
                    raise RuntimeError(
                        "Time step became too small. Model stiff/diverged."
                    )

                # Restore previous values to try again
                for s_obj in species_struct:
                    s_obj["var"].value[:] = last_sol_backup[s_obj["name"]]

        # E. Calculate Change (Steady State Check)
        max_change = 0.0
        for s_obj in species_struct:
            diff = np.max(np.abs(s_obj["var"].value - last_sol_backup[s_obj["name"]]))
            max_change = max(max_change, diff)

        total_time += current_dt

        # F. Adapt Time Step based on convergence trend
        if step > 1:
            if max_change < last_max_change:
                current_dt = min(current_dt * growth_factor, dt_max)
            else:
                current_dt = max(current_dt * cut_factor, mp.dt_min)
        else:
            # For the first step, use the old threshold logic
            if max_change < mp.dt_tolerance:
                current_dt = min(current_dt * growth_factor, dt_max)

        last_max_change = max_change

        # Reporting
        if step % mp.report_step == 0:
            if step > 0:
                print(
                    f"Time: {get_time_units(total_time):.2f~P}, "
                    f"dt: {get_time_units(current_dt):.2f~P}, "
                    f"Max Change: {max_change:.2e}"
                )
                df, fqfn = save_data_async(
                    mp,
                    c,
                    k,
                    species_list_full,
                    z,
                    D_mol,
                    diagenetic_reactions,
                    current_dt,
                )
        # Check for Steady State
        if max_change < mp.tolerance:
            print("Steady State Criteria Met.")
            break

    # 3. Final Report
    status = "Converged" if step < mp.max_steps else "Failed"
    print(f"{status} in {step} steps. Wall time: {time.time() - start_wall:.2f}s")

    df, fqfn = save_data_async(
        mp, c, k, species_list_full, z, D_mol, diagenetic_reactions, current_dt
    )
    print(
        f"Solver finished in {step} steps. Wall time: {time.time() - start_wall:.2f}s"
    )
    return step, max_change


def run_steady_state_solver_coupled(
    mp,
    c,
    species_list_full,
    species_list_partial,
    k,
    diagenetic_reactions,
    mesh,
    D_mol,
    bc_map,
    z,
):
    """
    Solve the equation system as a set of coupled reactions, rather than
    sequentially.
    """
    from fipy import LinearLUSolver, CellVariable
    from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
    from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
    from fipy.terms.diffusionTerm import DiffusionTerm
    import numpy as np
    import time
    from functools import reduce

    start_wall = time.time()
    print("Starting Coupled Steady State Solver...")
    solver = LinearLUSolver(tolerance=mp.tolerance)

    # 1. PRE-BUILD THE EQUATIONS
    # --------------------------
    lhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}
    rhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}

    # Store cross-term placeholders: dict of dicts
    # cross_vars[target_species][source_species] = CellVariable
    cross_vars = {s: {} for s in species_list_partial}

    # Analyze Topology once
    f_init, RATES = diagenetic_reactions(mp, c, k, f=data_container())

    for s in species_list_partial:
        res = getattr(f_init, s)
        if len(res) > 3:
            # res[3] is the list of (source_name, coeff) couplings
            for source_name, _ in res[3]:
                if source_name not in cross_vars[s]:
                    cross_vars[s][source_name] = CellVariable(mesh=mesh, value=0.0)

    eqs = []

    for species_name in species_list_partial:
        var = getattr(c, species_name)
        props = bc_map[species_name]

        # -- Transport --
        D_total = getattr(D_mol, species_name) + D_mol.D_bio
        D_total = np.maximum(D_total, 1e-20)

        vel = mp.w
        if props["type"] == "dissolved":
            vel = mp.w - mp.advection

        u_var = CellVariable(mesh=mesh, value=([vel],), rank=1)
        # FIX: Explicitly bind variables to terms for coupled solver inference
        conv_term = PowerLawConvectionTerm(coeff=u_var, var=var)
        diff_term = DiffusionTerm(coeff=CellVariable(mesh=mesh, value=D_total), var=var)

        # -- Reactions (Diagonal) --
        lhs_term = ImplicitSourceTerm(coeff=lhs_vars[species_name], var=var)
        rhs_term = rhs_vars[species_name]

        # -- Reactions (Cross-Coupling) --
        # Add implicit source terms for dependencies on other variables
        cross_reaction_terms = 0.0
        for source_name, cv in cross_vars[species_name].items():
            source_var = getattr(c, source_name)
            # ImplicitSourceTerm(coeff, var) adds `coeff * var` to the equation check.
            # In our convention: dC/dt = Rate. If Rate = k * Source.
            # We want + k*Source.
            # ImplicitSourceTerm(coeff=k, var=Source) adds +k*Source to the Operator?
            # FiPy equation: ... == ... + term.
            # Yes.
            cross_reaction_terms += ImplicitSourceTerm(coeff=cv, var=source_var)

        # -- Irrigation --
        if props["type"] == "dissolved":
            irr_sink = ImplicitSourceTerm(
                coeff=-CellVariable(mesh=mesh, value=D_mol.D_irr), var=var
            )
            irr_source = CellVariable(mesh=mesh, value=D_mol.D_irr * props["top"])
        else:
            irr_sink = 0.0
            irr_source = 0.0

        # Assemble Equation
        eq = (
            conv_term
            == diff_term
            + lhs_term
            + rhs_term
            + cross_reaction_terms
            + irr_sink
            + irr_source
        )
        eqs.append(eq)

    # Create the Coupled Equation System
    coupled_eq = reduce(lambda a, b: a & b, eqs)

    # 2. PICARD ITERATION LOOP
    # ------------------------
    dt_large = 1e10  # Large time step for steady state equilibrium checks
    max_change = 1e10
    step = 0

    while max_change > mp.tolerance and step < mp.max_steps:
        step += 1

        last_sol = {s: getattr(c, s).value.copy() for s in species_list_full}

        # Update reaction terms
        f_res, RATES = diagenetic_reactions(mp, c, k, f=data_container())

        # update matrix
        for species_name in species_list_partial:
            res_tuple = getattr(f_res, species_name)
            lhs_val = res_tuple[0]
            rhs_val = res_tuple[1]

            # Update Diagonal
            lhs_vars[species_name].setValue(getattr(lhs_val, "value", lhs_val))
            rhs_vars[species_name].setValue(getattr(rhs_val, "value", rhs_val))

            # Update Cross-Terms if present
            if len(res_tuple) > 3:
                # Accumulate coefficients first to avoid overwriting if multiple terms exist for same source
                # Initialize with 0.0
                batch_coeffs = {src: 0.0 for src in cross_vars[species_name]}

                for source_name, coeff in res_tuple[3]:
                    if source_name in batch_coeffs:
                        val = getattr(coeff, "value", coeff)
                        batch_coeffs[source_name] += val

                # Update CellVariables
                for src_name, total_coeff in batch_coeffs.items():
                    target_cv = cross_vars[species_name][src_name]
                    target_cv.setValue(total_coeff)

        # Sweep the Coupled System
        res = coupled_eq.sweep(solver=solver)

        # Relaxation and Convergence
        max_change = 0
        for species_name in species_list_partial:
            var = getattr(c, species_name)
            new_val = relax_solution(var.value, last_sol[species_name], mp.relax)
            var.setValue(new_val)

            change = np.max(np.abs(var.value - last_sol[species_name]))
            max_change = max(max_change, change)

        if step % mp.report_step == 0:
            if step > 0:
                print(
                    f"Iteration {step}: Max Var Change {max_change:.2e}, Coupled Residual {res:.2e}"
                )
                df, fqfn = save_data_async(
                    mp,
                    c,
                    k,
                    species_list_full,
                    z,
                    D_mol,
                    diagenetic_reactions,
                    dt_large,
                )

    # 3. FINALIZE
    if step >= mp.max_steps:
        converged = "No"
        print(
            f"Warning: Coupled solver did not converge. Last change: {max_change:.2e}"
        )
    else:
        converged = "Yes"
        print(f"Coupled steady state converged in {step} iterations.")

    end_wall = time.time()
    total_time = end_wall - start_wall
    print(f"Coupled Solver Wall Time: {total_time:.2f} seconds")

    return converged, step, total_time


def run_non_steady_state_solver_coupled_bdf(
    mp,
    c,
    species_list_full,
    species_list_partial,
    k,
    diagenetic_reactions,
    mesh,
    D_mol,
    bc_map,
    z,
):
    """
    Solve the non-steady state coupled system using Scipy's BDF solver.

    This treats the system as a large ODE system: dC/dt = F(C, t).
    F(C) includes diffusion, advection, and reactions.
    """
    from scipy.integrate import solve_ivp
    from fipy import CellVariable
    from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
    from fipy.terms.diffusionTerm import DiffusionTerm
    from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
    from functools import reduce
    from reactions_new import equilibrium_reactions
    import numpy as np

    print("Starting BDF Solver (scipy.integrate.solve_ivp)...")

    # 1. SETUP EQUATIONS (Spatial Terms Only)
    # ---------------------------------------
    # We construct the spatial part of the operator: F(C) = Diffusion - Advection + Reaction
    # Note: Fipy's equation notation usually implies LHS == RHS.
    # To get dC/dt = F(C), we construct 'eq' such that F(C) is the residual of 'eq'.
    # Standard: TransientTerm == Diff + React - Conv
    # Rearranged: dC/dt = Diff + React - Conv
    # So we sum these terms.

    lhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}
    rhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}
    cross_vars = {s: {} for s in species_list_partial}

    # Analyze Topology once to setup cross-term variables
    f_init, RATES = diagenetic_reactions(mp, c, k, f=data_container())
    for s in species_list_partial:
        res = getattr(f_init, s)
        if len(res) > 3:
            for source_name, _ in res[3]:
                if source_name not in cross_vars[s]:
                    cross_vars[s][source_name] = CellVariable(mesh=mesh, value=0.0)

    # Optimization: Cache objects
    species_struct = []
    spatial_eqs = []

    for species_name in species_list_partial:
        var = getattr(c, species_name)
        props = bc_map[species_name]

        # Cache
        species_struct.append(
            {
                "name": species_name,
                "var": var,
                "lhs": lhs_vars[species_name],
                "rhs": rhs_vars[species_name],
                "cross": cross_vars[species_name],
            }
        )

        # -- Transport --
        D_total = np.maximum(getattr(D_mol, species_name) + D_mol.D_bio, 1e-20)
        vel = mp.w - mp.advection if props["type"] == "dissolved" else mp.w

        u_var = CellVariable(mesh=mesh, value=([vel],), rank=1)
        conv_term = PowerLawConvectionTerm(coeff=u_var, var=var)
        diff_term = DiffusionTerm(coeff=CellVariable(mesh=mesh, value=D_total), var=var)

        # -- Reactions --
        # Implicit Source Term (LHS diagonal)
        lhs_term = ImplicitSourceTerm(coeff=lhs_vars[species_name], var=var)
        rhs_term = rhs_vars[species_name]

        # Cross Terms
        cross_terms = 0.0
        for src_name, cv in cross_vars[species_name].items():
            src_var = getattr(c, src_name)
            cross_terms += ImplicitSourceTerm(coeff=cv, var=src_var)

        # -- Irrigation --
        irr_term = 0.0
        if props["type"] == "dissolved":
            irr_term = ImplicitSourceTerm(
                coeff=CellVariable(mesh=mesh, value=-D_mol.D_irr), var=var
            ) + CellVariable(mesh=mesh, value=D_mol.D_irr * props["top"])

        # Assembly: dC/dt = Diff + Sources - Conv
        # We sum them up. Fipy `residual` evaluates (LHS - RHS) usually.
        # But here we are just adding terms to an expression object.
        # Warning: PowerLawConvectionTerm is typically LHS.
        # Term signs in Fipy:
        # Eq: Transient == Diffusion + Source - Convection
        # If we treat them as expressions:
        # RHS_Expression = diff_term + lhs_term + rhs_term + cross_terms + irr_term - conv_term
        # Note: 'lhs_term' here is actually a Source (Reaction Rate coefficient).
        # Typically Rate = k * C. ImplicitSourceTerm(coeff=k, var=C) evaluates to k*C.

        eq_expr = diff_term + lhs_term + rhs_term + cross_terms + irr_term - conv_term
        spatial_eqs.append(eq_expr)

    # Combine into one coupled system
    coupled_eq_spatial = reduce(lambda a, b: a & b, spatial_eqs)

    # 2. DEFINE ODE FUNCTION
    # ----------------------
    c_vars = [s["var"] for s in species_struct]

    last_y = [None]

    def update_state(y):
        """Update Fipy variables and reaction coefficients from solver state y."""
        # Simple cache: check if y has changed significantly
        if last_y[0] is not None and np.allclose(y, last_y[0], atol=1e-15, rtol=1e-15):
            return
        last_y[0] = y.copy()

        # 1. Update Fipy Variables
        offset = 0
        for v in c_vars:
            n = v.mesh.numberOfCells
            v.value[:] = y[offset : offset + n]
            offset += n

        # 2. Update Reaction Coefficients
        f_res, _ = diagenetic_reactions(mp, c, k, f=data_container())

        # Update Matrix Coefficients (LHS/RHS of the reaction terms)
        for s_obj in species_struct:
            name = s_obj["name"]
            res_tuple = getattr(f_res, name)

            # Update Diagonal
            lhs_val = res_tuple[0]
            if hasattr(lhs_val, "value"):
                lhs_val = lhs_val.value
            s_obj["lhs"].setValue(lhs_val)  # coeff

            rhs_val = res_tuple[1]
            if hasattr(rhs_val, "value"):
                rhs_val = rhs_val.value
            s_obj["rhs"].setValue(rhs_val)  # rhs

            # Update Cross-Terms
            if len(res_tuple) > 3:
                # Reset accumulators
                batch_coeffs = {src: 0.0 for src in s_obj["cross"]}
                for source_name, coeff in res_tuple[3]:
                    if source_name in batch_coeffs:
                        val = getattr(coeff, "value", coeff)
                        batch_coeffs[source_name] += val

                for src, val in batch_coeffs.items():
                    s_obj["cross"][src].setValue(val)

    from diff_lib import get_time_units

    last_reported_t = [0.0]

    def f(t, y):
        update_state(y)
        # 3. Calculate Residual
        rhs = coupled_eq_spatial.justResidualVector()
        return rhs

    def jac(t, y):
        update_state(y)

        # Build the matrix using internal API
        # _prepareLinearSystem returns a Solver object that contains the matrix
        solver = coupled_eq_spatial._prepareLinearSystem(
            var=None, solver=None, boundaryConditions=(), dt=None
        )
        m_fipy = solver.matrix

        # Scipy's BDF solver requires a numpy-compatible matrix (dense or scipy.sparse)
        # FiPy's .numpyArray property provides this regardless of the backend (Scipy/PETSc)
        if hasattr(m_fipy, "numpyArray"):
            return m_fipy.numpyArray
        return m_fipy

    # 3. SOLVE
    # --------
    from scipy.integrate import BDF
    # Initial Condition
    y0 = np.concatenate([v.value.ravel() for v in c_vars])

    solver = BDF(
        f,
        0,
        y0,
        mp.t_end,
        jac=jac,
        rtol=mp.tolerance,
        atol=1e-12,
        first_step=mp.dt_min,
        max_step=mp.dt_max,
    )

    print(f"  Starting manual stepping loop...")
    
    last_reported_t = 0.0
    step_count = 0

    while solver.status == "running":
        solver.step()
        step_count += 1
        t = solver.t
        dt = solver.step_size

        # throttle reporting to once every ~1% of simulation time or every 5 seconds wall time
        if t - last_reported_t > mp.t_end * 0.01 or t >= mp.t_end:
            print(
                f"Progress {t / mp.t_end * 100:6.2f}%, "
                f"t = {get_time_units(t):.2f~P}, "
                f"dt_solver = {get_time_units(dt):.2f~P}"
            )
            update_state(solver.y) # Ensure variables are synced for saving
            save_data_async(
                mp,
                c,
                k,
                species_list_full,
                z,
                D_mol,
                diagenetic_reactions,
                dt,
            )
            last_reported_t = t

    # 4. UPDATE FINAL STATE
    # ---------------------
    final_y = solver.y
    offset = 0
    for v in c_vars:
        n = v.mesh.numberOfCells
        v.value[:] = final_y[offset : offset + n]
        offset += n

    # Report
    print(f"BDF Solver finished: solver.status={solver.status}.")
    print(f"  Total function evaluations (nfev): {solver.nfev}")
    print(f"  Total Jacobian evaluations (njev): {solver.njev}")
    print(f"  Total LU decompositions (nlu): {solver.nlu}")

    return solver.nfev, 0.0
