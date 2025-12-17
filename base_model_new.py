"""FiPy pyrite burial base model.

This module replaces the fastfd architecture with FiPy.
"""

import fipy as fp
import numpy as np
from diff_lib import make_grid, data_container


def model_setup(mp, k, bc, od, c, reaction_rates, diagenetic_reactions, results):
    """Set up and run the FiPy model."""

    # 1. Grid Generation
    # Use variable grid if requested implicitly by the need for stability,
    # or just use the helper.
    # We assume 'mp' might theoretically carry info, but we use defaults/hardcoded 1mm surface for now.
    initial_spacing = 0.001
    if hasattr(mp, "initial_spacing"):
        initial_spacing = mp.initial_spacing

    mesh, z_centers = make_grid(mp.max_depth, mp.grid_points, initial_spacing)

    # 2. Variables Setup
    # 'c' initially contains numpy arrays (zeros). We upgrade them to CellVariables.
    # We maintain the property that 'c' attributes are accessible.

    vars_dict = {}
    for v_name in vars(c):
        # We initialize with the values from 'c' (which are essentially initial guesses/conditions)
        # However, 'c' arrays are length N (from run_new). mesh is length N.
        val = getattr(c, v_name)

        # Create CellVariable
        var = fp.CellVariable(name=v_name, mesh=mesh, value=val, hasOld=True)
        vars_dict[v_name] = var
        setattr(c, v_name, var)

    # 'sc' in fastfd was used for scalar access. In FiPy, 'c' holds the variables.
    # We pass 'c' as 'sc' to reactions.
    sc = c

    # 3. Boundary Conditions
    for v_name, var in vars_dict.items():
        if v_name not in bc:
            continue

        specs = bc[v_name]
        # specs: [top_type, top_val, bot_type, bot_val, phase, type]

        top_type, top_val = specs[0], specs[1]
        bot_type, bot_val = specs[2], specs[3]

        # Top BC (Faces Left in 1D)
        if top_type == "concentration":
            var.constrain(top_val, mesh.facesLeft)
        elif top_type == "gradient":
            var.faceGrad.constrain(top_val, mesh.facesLeft)

        # Bottom BC (Faces Right)
        if bot_type == "concentration":
            var.constrain(bot_val, mesh.facesRight)
        elif bot_type == "gradient":
            var.faceGrad.constrain(bot_val, mesh.facesRight)

    # 4. Coefficients re-mapping
    # 'od' contains diffusion coefficients and parameters.
    # IN fastfd/run_new, these were created on a linear grid.
    # If we are strictly correcting grid issues, we should ideally re-calculate them on new z_centers.
    # But 'od' is passed in.
    # For robust migration, we will check if 'od' attributes are numpy arrays.
    # If they are length N, we assume they match index-wise (mapping linear index to stretched index).
    # This distorts the physics slightly (stretching the profile), but is physically valid if we consider
    # the input profile was just "generic shape".
    # However, 'od.z' is present. We can interpolate if we want accuracy!
    # Let's try to interpolate 'od' values to new grid if 'od.z' exists.

    if hasattr(od, "z"):
        old_z = od.z
        new_z = z_centers
        for attr, val in vars(od).items():
            if isinstance(val, np.ndarray) and len(val) == len(old_z) and attr != "z":
                # Interpolate
                setattr(od, attr, np.interp(new_z, old_z, val))
        # Update od.z
        od.z = new_z

    # 5. Build Transport Terms (Constant)
    eqn_base = {}

    for v_name, var in vars_dict.items():
        if v_name not in bc:
            continue
        specs = bc[v_name]
        phase = specs[4]

        # Diffusion Coefficient
        if phase == "dissolved":
            D_mol = getattr(od, v_name)
            D_bio = od.DB
            D_total = D_mol + D_bio
            w_val = od.w - mp.advection
        else:
            D_total = od.DB
            w_val = od.w

        # Convection Velocity
        # In FiPy 1D, convection coeff is vector (w,).
        # w_val is array.
        # We create a CellVariable (rank 1) for velocity.
        u_var = fp.CellVariable(mesh=mesh, rank=1)
        u_var[0] = w_val

        # Wrap D_total in CellVariable to avoid shape ambiguity
        D_var = fp.CellVariable(mesh=mesh, value=D_total, name=f"D_{v_name}")

        # Transport Equation: Diffusion - Convection
        trans = fp.DiffusionTerm(coeff=D_var) - fp.ConvectionTerm(coeff=u_var)

        # Irrigation (Bioturbation Type 2 / Non-local exchange)
        # - BI * C (Implicit Sink) + BI * C_bound (Explicit Source)
        # Note: 'specs[1]' is used as boundary value for irrigation source usually.
        if hasattr(od, "BI"):
            # Wrap BI to ensure Rank 0
            BI_var = fp.CellVariable(mesh=mesh, value=od.BI, name="BI_coeff")
            irr_sink = fp.ImplicitSourceTerm(coeff=-BI_var)
            # Source: BI * C_top
            # Explicit source is just added to the equation
            irr_src = BI_var * specs[1]
            trans += irr_sink + irr_src

        eqn_base[v_name] = trans

    # 6. Solution Loop
    # We iterate to solve the coupled non-linear system.

    # Initialize 'f' container for reporting
    # f_res = data_container(" ".join(vars_dict.keys()), np.zeros(mp.grid_points))

    tolerance = mp.res_min

    print(f"Starting FiPy solve for {mp.max_loops} iterations...")

    for iteration in range(mp.max_loops):
        # Store old values for relaxation/convergence
        for v in vars_dict.values():
            v.updateOld()

            f_out, alpha = diagenetic_reactions(
                mp, z_centers, c, k, reaction_rates, c, results
            )

        max_resid = 0.0

        # Solve Equations
        for v_name, var in vars_dict.items():
            if v_name not in eqn_base:
                continue

            lhs, rhs, rate_val = getattr(f_out, v_name)

            # Safety: Ensure LHS/RHS are treated as Rank 0 Variables if they are numpy arrays
            # (e.g. rhs_poc is np.zeros_like(...))
            if isinstance(lhs, np.ndarray):
                lhs = fp.CellVariable(mesh=mesh, value=lhs, name=f"LHS_{v_name}")
            if isinstance(rhs, np.ndarray):
                rhs = fp.CellVariable(mesh=mesh, value=rhs, name=f"RHS_{v_name}")

            # Use direct addition for explicit source
            eq = eqn_base[v_name] + fp.ImplicitSourceTerm(coeff=lhs) - rhs

            # Sweep
            # We use a large dt to simulate steady state step, or just sweep.
            # Using relax factor manually.

            res = eq.sweep(var=var, dt=1e6)

            # Relaxation
            relax = mp.relax
            var.setValue(var.value * relax + var.old.value * (1 - relax))

            # Check convergence (relative change or simple diff)
            # using 'residual' metric from run_new: abs(sum(diff)) / N
            # But run_new calculated it on 'so4' mostly.

            diff = np.abs(var.value - var.old.value)
            crs = np.sum(diff) / mp.grid_points
            if v_name == "so4":
                max_resid = crs

        if iteration % 10 == 0:
            print(f"Iter {iteration}: so4 resid = {max_resid:.2e}")

        if max_resid < tolerance and iteration > 10:
            print(f"Converged at iter {iteration} with resid {max_resid:.2e}")
            break

    # Pack results
    # c is already updated.
    # reaction_rates (f_out) contains tuples of variables/arrays.
    # run_new expects f_dict to contain values.
    # We should convert the Variables in f_out to numpy arrays.

    # f_out is 'reaction_rates'.
    # We iterate and value-ify.
    for v_name in vars(f_out):
        val = getattr(f_out, v_name)
        # val is (LHS, RHS, RATES). RATES is what we usually plot/export.
        # RATES is 3rd element.
        # run_new: "data = getattr(reaction_rates, v); f_dict[v] = np.abs(data[2])"
        # So we just need to ensure data[2] is numpy array.
        # If it is a FiPy variable arithmetic result, we need .value?

        lhs, rhs, rates = val
        # Check if they are Variables
        if hasattr(rates, "value"):
            rates = rates.value
        if hasattr(lhs, "value"):
            lhs = lhs.value
        if hasattr(rhs, "value"):
            rhs = rhs.value

        setattr(f_out, v_name, (lhs, rhs, rates))

    return od, c, f_out
