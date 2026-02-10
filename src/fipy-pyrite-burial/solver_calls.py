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
    from fipy import LinearLUSolver, CellVariable
    from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
    from fipy.terms.diffusionTerm import DiffusionTerm
    from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
    from fipy.terms.transientTerm import TransientTerm
    from functools import reduce
    from reactions_new import equilibrate_fes_precipitation

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

    solver = LinearLUSolver(tolerance=mp.tolerance)

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
                # Note: We pass the CURRENT guess 'c' to reaction function
                f_res, RATES = diagenetic_reactions(mp, c, k, f=data_container())

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

                # D. Apply Instantaneous Equilibrium (Operator Splitting)
                # This modifies c.fe2, c.h2s, c.fes IN PLACE
                RATES = equilibrate_fes_precipitation(c, k, mp, current_dt, RATES)

                # If we get here without error, the linear solve worked.
                step_converged = True

            except Exception as e:
                # If solver diverged or crashed
                print(
                    f"  Step failed at dt={current_dt:.1e}: {e}. Retrying with smaller dt."
                )
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
                mp, c, k, species_list_full, z, D_mol, diagenetic_reactions, current_dt
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
