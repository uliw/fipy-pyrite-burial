"""
Utility library for the /fastfd/ package.

This package provides a small collection of functions and a lightweight container class
used throughout the Pyrite Burial model.

The module contains:

    - :class:=data_container= – a simple container that can be initialised from a
      space‑separated string of attribute names with optional default values, or from a
      dictionary mapping attribute names to values.

    - :func:=diff_coeff= – computes the diffusion coefficient (m² s⁻¹) for a given
      temperature (°C), porosity (percent) and the linear parameters /m0/ and /m1/ from
      Boudreau (1996).

    - :func:=get_delta= – calculates the isotopic delta value (‰) from the total
      concentration of an isotope pair and a reference ratio.

    - :func:=get_l_mass= – derives the concentration of the light isotope from a
      measured total concentration, a delta value and the reference ratio.

    - :func:=relax_solution= – blends a current solution vector with a previous one,
      limiting the change to a specified fraction and enforcing non‑negative values.

These helpers are primarily intended for modelling isotope diffusion and fractionation
processes in geological simulations.
"""

import numpy as np
from typing import Union


class data_container(dict):
    """A dictionary-based container with attribute access.

    Supports initialization from a space-separated string or a dictionary.
    """

    def __init__(self, names=None, defaults=None):
        super().__init__()
        if isinstance(names, str):
            names = names.split(" ")
            if isinstance(defaults, list):
                for i, name in enumerate(names):
                    if name != "":
                        self[name] = defaults[i]
            else:
                for name in names:
                    if name != "":
                        self[name] = defaults
        elif isinstance(names, dict):
            self.update(names)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )


def diff_coeff(T, m0, m1, phi):
    """Calculate the diffusion coeefficien in m^2/s.

    T: temperature in C
    phi: porosity in percent
    m0, m1: parameter as from table X in Boudreau 1996
    """
    return (m0 + m1 * T) * 1e-10 / (1 - np.log(phi**2))


def get_delta(c, li, r):
    """Calculate the delta from the mass of light and heavy isotope.

    :param li: light isotope mass/concentration
    :param h: heavy isotope mass/concentration
    :param r: reference ratio

    :return : delta

    """
    h = c - li

    return np.where(li < 0.001, float("nan"), 1000 * (h / li - r) / r)


def get_delta_from_concentration(c, li, r):
    """Calculate the delta from the mass of light and heavy isotope.

    :param c: total mass/concentration
    :param l: light isotope mass/concentration
    :param r: reference ratio

    """
    h = c - li
    d = 1000 * (h / li - r) / r
    return d


def get_l_mass(m, d, r):
    """Derive the concentration of the light isotope.

    From a measured total concentration, a delta value and the reference ratio.
    :param m: mass or concentration
    :param d: delta value
    :param r: isotopic reference ratio

    return mass or concentration of the light isotopeb
    """
    return (1000.0 * m) / ((d + 1000.0) * r + 1000.0)


def relax_solution(curr_sol, last_sol, fraction):
    """Blend two solution vectors.

    In such away that they only chance by a given fraction
    """
    sol = last_sol * (1 - fraction) + curr_sol * fraction
    return sol * (sol >= 0)  # exclude negative solutions


def bioturbation_profile(z, D_max, cutoff_depth, threshold=1e-12):
    """
    Calculate the necessary steepness.

    To fit a mixing profile into 'cutoff_depth', ensuring D_max at surface and
    'threshold' at cutoff.
    """
    # 1. Safety Checks
    if cutoff_depth <= 0 or D_max <= threshold:
        return np.zeros_like(z)

    # 2. Calculate the Magnitude of the Drop
    # We need to drop from D_max down to threshold.
    # ratio represents the magnitude of this drop (e.g., 10,000,000x)
    ratio = (D_max / threshold) - 1
    log_ratio = np.log(ratio)

    # 3. Dynamic Steepness Calculation
    # We want the "slide" to start at z=0 and finish exactly at z=cutoff_depth.
    # To fit the full drop (log_ratio) into the distance (cutoff_depth):
    # k = total_drop_in_log_units / distance
    # We add a safety factor (e.g., 1.1) to ensure the 'shoulder' is slightly below surface
    calculated_steepness = (log_ratio / cutoff_depth) * 1.05

    # 4. Calculate Inflection Point
    # The inflection point is where the value is half of D_max.
    # Based on the calculated steepness, we shift the curve so the tail hits
    # 'threshold' exactly at 'cutoff_depth'.
    shift = log_ratio / calculated_steepness
    inflection_point = cutoff_depth - shift

    # 5. Generate Sigmoid
    sigmoid = D_max / (1 + np.exp(calculated_steepness * (z - inflection_point)))

    # 6. Hard Clamp
    sigmoid[z > cutoff_depth] = 0.0

    return sigmoid


def bioturbation_profile_2(z, D_max, cutoff_depth, threshold=1e-12):
    """
    Generate a sigmoid mixing profile.

    That fits EXACTLY within 'cutoff_depth'.  Calculates dynamic steepness so shallow
    depths get sharper curves automatically.
    """
    # Safety: If depth is zero or D_max is negligible, return zeros
    if cutoff_depth <= 0 or D_max <= threshold:
        return np.zeros_like(z)

    # 1. Calculate how "steep" we need to be to drop from D_max to threshold
    #    within the allocated depth.
    #    formula: D_max / (1 + exp(k * z)) = threshold
    ratio = (D_max / threshold) - 1
    if ratio <= 0:
        return np.zeros_like(z)

    total_drop_log = np.log(ratio)

    # We want the drop to finish slightly before the cutoff (95% of depth)
    # to ensure the clamp is clean.
    effective_depth = cutoff_depth * 0.95
    calculated_steepness = total_drop_log / effective_depth

    # 2. Calculate the "Shift" (Inflection Point)
    #    We shift the curve so it starts flat at surface and drops at the end.
    #    Shifting by log(ratio)/k moves the tail to the cutoff.
    #    We shift slightly less to keep the 'shoulder' near the surface.
    shift = total_drop_log / calculated_steepness
    inflection_point = cutoff_depth - shift

    # 3. Generate Profile
    sigmoid = D_max / (1 + np.exp(calculated_steepness * (z - inflection_point)))

    # 4. Hard Clamp to ensure true zero below the cutoff
    sigmoid[z > cutoff_depth] = 0.0

    return sigmoid


def make_grid(L, N, initial_spacing):
    """
    Construct a 1D grid with variable spacing (geometric progression).

    The grid is finer near z=0 and coarser at depth.

    Args:
    -----
        L (float): Total length of the domain (max depth).
        N (int): Number of cells.
        initial_spacing (float): The size of the first cell (at z=0).

    Returns:
    -------
        tuple (mesh, z_centers)
            mesh: A fipy.Grid1D object.
            z_centers: A numpy array of cell center coordinates.
    """
    from fipy import Grid1D
    import numpy as np
    from scipy.optimize import brentq

    # Case 1: Uniform Grid is sufficient or requested spacing is large
    if abs(initial_spacing * N - L) < 1e-9:
        dx = L / N
        return Grid1D(nx=N, dx=dx), np.linspace(dx / 2, L - dx / 2, N)

    if initial_spacing * N > L:
        # Fallback to uniform if initial spacing is too large
        print("Warning: initial_spacing * N > L. Reverting to uniform grid.")
        dx = L / N
        return Grid1D(nx=N, dx=dx), np.linspace(dx / 2, L - dx / 2, N)

    # Case 2: Geometric Progression
    # Sum = a * (r^N - 1) / (r - 1) = L
    # We look for r > 1.
    # Function to zero: a * (r^N - 1) / (r - 1) - L
    # But to avoid division by zero near 1, we can use the multiplied form but search strictly > 1.

    def func(r):
        return initial_spacing * (r**N - 1) - L * (r - 1)

    # Bracket search.
    # r=1 produces a*N - L < 0 (since a*N < L)
    # r=2 produces massive number.
    # Root is between 1+epsilon and 2 (usually very close to 1).

    try:
        r_solution = brentq(func, 1.00000001, 2.0)
    except Exception as e:
        print(f"Grid generation optimization failed: {e}. Reverting to linear.")
        dx = L / N
        return Grid1D(nx=N, dx=dx), np.linspace(dx / 2, L - dx / 2, N)

    # Generate faces
    faces = np.zeros(N + 1)
    current_dx = initial_spacing

    faces[0] = 0
    for i in range(1, N + 1):
        faces[i] = faces[i - 1] + current_dx
        current_dx *= r_solution

    # Force the last one to be exactly L, but check error
    if abs(faces[-1] - L) > 1e-3:
        # If divergence is high, something went wrong, but usually we just clip.
        pass
    faces[-1] = L

    # Calculate dx array for fipy
    dx_array = np.diff(faces)

    mesh = Grid1D(dx=dx_array)
    z_centers = mesh.cellCenters[0].value

    return mesh, z_centers


def run_steady_state_solver(
    mp,
    equations,
    c,
    species_list,
    k,
    diagenetic_reactions,
    mesh,
    D_mol,
    D_bio,
    D_irr,
    bc_map,
):
    """
    Run the FiPy solver loop for steady state with Picard iteration for non-linearity.
    """
    from fipy import LinearLUSolver, CellVariable
    import numpy as np
    import time

    start_wall = time.time()
    print("Starting Steady State Solver (Coupled Picard Iteration)...")
    solver = LinearLUSolver(tolerance=mp.tolerance)

    max_change = 1e10
    step = 0

    # Pre-build the transport part of the equations (they are linear and constant)
    # Re-use build_steady_state_equations logic but only for transport
    transport_eqs = {}
    for species_name in species_list:
        var = getattr(c, species_name)
        props = bc_map[species_name]

        D_total = getattr(D_mol, species_name) + D_bio
        vel = mp.w
        if props["type"] == "dissolved":
            vel = mp.w - mp.advection

        # Scaling factor for porosity
        phi = mp.phi
        scaling = phi if props["type"] == "dissolved" else (1.0 - phi)

        # Divided form: no theta scaling for convection and diffusion coefficients
        u_var = CellVariable(mesh=mesh, value=([vel * scaling],), rank=1)
        from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
        from fipy.terms.diffusionTerm import DiffusionTerm

        conv_term = PowerLawConvectionTerm(coeff=u_var)
        diff_term = DiffusionTerm(
            coeff=CellVariable(mesh=mesh, value=D_total * scaling)
        )
        transport_eqs[species_name] = (conv_term, diff_term)

    while max_change > mp.tolerance and step < mp.max_steps:
        step += 1

        # Store previous iteration values for relaxation and convergence check
        last_sol = {s: getattr(c, s).value.copy() for s in species_list}

        # 1. Update reaction terms based on CURRENT concentration values
        f_res = diagenetic_reactions(mp, c, k, f=data_container())

        res = 0
        total_change = 0

        from fipy.terms.implicitSourceTerm import ImplicitSourceTerm

        for species_name in species_list:
            var = getattr(c, species_name)
            props = bc_map[species_name]

            # Reconstruct the equation with UPDATED reaction terms
            conv_term, diff_term = transport_eqs[species_name]

            lhs_val = getattr(f_res, species_name)[0]
            rhs_val = getattr(f_res, species_name)[1]

            lhs_term = ImplicitSourceTerm(coeff=lhs_val * scaling)

            if hasattr(rhs_val, "rank"):
                rhs_term = -rhs_val * scaling
            else:
                rhs_term = CellVariable(mesh=mesh, value=-rhs_val * scaling)

            # Irrigation only affects dissolved species
            if props["type"] == "dissolved":
                irr_sink = ImplicitSourceTerm(
                    coeff=-CellVariable(mesh=mesh, value=D_irr * mp.phi)
                )
                irr_source = CellVariable(
                    mesh=mesh, value=D_irr * props["top"] * mp.phi
                )
            else:
                irr_sink = 0.0
                irr_source = 0.0

            eq = conv_term == diff_term + lhs_term + rhs_term + irr_sink + irr_source

            # Sweep to update the variable
            res += eq.sweep(var=var, solver=solver)

        # 2. Apply relaxation and calculate max relative change
        max_change = 0
        for species_name in species_list:
            var = getattr(c, species_name)

            # Relaxation
            new_val = relax_solution(var.value, last_sol[species_name], mp.relax)
            var.setValue(new_val)

            # Convergence check: max change relative to scale
            # We use absolute change here but could use relative if values are large
            change = np.max(np.abs(var.value - last_sol[species_name]))
            # Scale by typical value if needed, for now just max absolute
            max_change = max(max_change, change)

        if step % 10 == 0 or step == 1:
            print(
                f"Iteration {step}: Max Var Change {max_change:.2e}, Linear Residual {res:.2e}"
            )

    if step >= mp.max_steps:
        converged = "No"
        print(
            f"Warning: Steady state solver did not converge after {mp.max_steps} steps. Last change: {max_change:.2e}"
        )
    else:
        converged = "Yes"
        print(
            f"Steady state converged in {step} iterations (Max change: {max_change:.2e})."
        )

    end_wall = time.time()
    total_time = end_wall - start_wall
    print(f"Steady State Wall Time: {total_change:.2f} seconds")

    return converged, step, total_time


def run_non_steady_solver(mp, equations, c, species_list):
    """
    Run the FiPy solver loop.
    """
    from fipy import LinearLUSolver, CellVariable
    import time

    start_wall = time.time()

    dt = 1e-2 * 3600 * 24 * 365.0  # Start with 0.01 years
    elapsed_time = 0.0
    target_time = mp.run_time * 3600 * 24 * 365.0  # 10,000 years

    print(f"Starting Solver... Target Time: {target_time / 3600 / 24 / 365:.1f} years")
    solver = LinearLUSolver(tolerance=mp.tolerance)

    step = 0
    while elapsed_time < target_time and step < mp.max_steps:
        step += 1

        # f_res, _ = diagenetic_reactions(mp, None, c, k, data_container(), None, None)
        # Update Old values
        for var in vars(c).values():
            if isinstance(var, CellVariable):
                var.updateOld()

        res = 1e10
        sweeps = 0
        while res > mp.tolerance and sweeps < 20:
            # Store previous iteration values for relaxation
            last_sol = {s: getattr(c, s).value.copy() for s in species_list}

            res = 0
            for var, eq in equations:
                res += eq.sweep(var=var, dt=dt, solver=solver)

            # Apply relaxation
            for species_name in species_list:
                var = getattr(c, species_name)
                var.setValue(
                    relax_solution(var.value, last_sol[species_name], mp.relax)
                )

            sweeps += 1

        elapsed_time += dt
        # Increase time step geometrically
        dt *= 1.2
        # Cap dt to prevent loss of accuracy
        if dt > mp.dt_max * 3600 * 24 * 365.0:
            dt = mp.dt_max * 3600 * 24 * 365.0

        if step % 10 == 0:
            print(
                f"Step {step}: Time {elapsed_time / 3600 / 24 / 365:.2f} yr (dt={dt / 3600 / 24 / 365:.2f} yr) - Residual {res:.2e}"
            )

    print("Simulation Complete.")
    end_wall = time.time()
    print(f"Non-Steady State Wall Time: {end_wall - start_wall:.2f} seconds")


def save_data(mp, c, k, species_list, z, D_mol, diagenetic_reactions):
    """
    Save the model results to a CSV file.
    """
    import pandas as pd
    import pathlib as pl

    f_final = diagenetic_reactions(mp, c, k, data_container())

    data = {"z": z}
    for species_name in species_list:
        data[f"c_{species_name}"] = getattr(c, species_name).value

    for species_name in species_list:
        rates_val = getattr(f_final, species_name)[2]
        if hasattr(rates_val, "value"):
            data[f"f_{species_name}"] = rates_val.value
        else:
            data[f"f_{species_name}"] = np.array(rates_val)

    # Save all items in D_mol
    for d_name, d_val in D_mol.items():
        if hasattr(d_val, "value"):
            data[d_name] = d_val.value
        else:
            data[d_name] = np.array(d_val)

    # calculate delta values
    for species_name in species_list:
        if "_32" in species_name:
            base_species = species_name[:-3]
            s = data[f"c_{base_species}"]
            """fes2_32 tracks the number of S-atoms, not molecules of FeS2.
            Since FeS2 tracks the number of molecules, fes2_32 is two times
            larger than fes2. To get the correct delta value, we beed to devide
            by two.
            """
            if species_name == "fes2_32":
                s32 = data[f"c_{species_name}"] / 2
            else:
                s32 = data[f"c_{species_name}"]
            data[f"d_{base_species}"] = get_delta(s, s32, mp.VCDT)

    data["w"] = np.ones(len(z)) * mp.w
    data["phi"] = np.ones(len(z)) * mp.phi

    df = pd.DataFrame(data)
    fqfn = pl.Path.cwd() / mp.plot_name
    df.to_csv(fqfn, index=False)
    print(f"Data saved to {fqfn}")

    return df, fqfn


def build_non_steady_equations(
    mp, c, k, species_list, mesh, D_mol, D_bio, D_irr, bc_map, diagenetic_reactions
):
    """
    Build the reaction-transport equations.
    """
    from fipy import CellVariable
    from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
    from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
    from fipy.terms.diffusionTerm import DiffusionTerm
    from fipy.terms.transientTerm import TransientTerm

    equations = []
    # Call reaction function once to set up the Terms
    # We pass None for z, sc, results as they might not be used or needed for setup
    f_res = diagenetic_reactions(mp, c, k, f=data_container())

    for species_name in species_list:
        var = getattr(c, species_name)
        props = bc_map[species_name]

        # 1. Transport
        D_total = getattr(D_mol, species_name) + D_bio

        vel = mp.w
        if props["type"] == "dissolved":
            vel = mp.w - mp.advection

        # Scaling for porosity
        phi = mp.phi
        scaling = phi if props["type"] == "dissolved" else (1.0 - phi)

        # Explicitly create Rank 1 CellVariable for velocity to avoid shape errors
        u_var = CellVariable(mesh=mesh, value=([vel * scaling],), rank=1)

        # Use VanLeerConvectionTerm to minimize numerical dispersion artifacts in isotope ratios
        # conv_term = VanLeerConvectionTerm(coeff=u_var)
        conv_term = PowerLawConvectionTerm(coeff=u_var)

        # Wrap D_total in CellVariable to avoid shape ambiguity
        diff_term = DiffusionTerm(
            coeff=CellVariable(mesh=mesh, value=D_total * scaling)
        )

        # 2. Reactions
        # Imported diagenetic_reactions returns (LHS_coeff, RHS_val, rate)
        lhs_val = getattr(f_res, species_name)[0]
        rhs_val = getattr(f_res, species_name)[1]

        # LHS from reactions_new is a constant or CellVariable expression.
        # We pass it directly to ImpicitSourceTerm to ensure it stays dynamic.
        lhs_term = ImplicitSourceTerm(coeff=lhs_val * scaling)

        # RHS from reactions_new is " - rate". We want "+ rate".
        # Use dynamic expression if it's a Variable, otherwise wrap static values
        if hasattr(rhs_val, "rank"):
            rhs_term = -rhs_val * scaling
        else:
            # It's an array or number (static)
            rhs_term = CellVariable(mesh=mesh, value=-rhs_val * scaling)

        # 3. Irrigation
        if props["type"] == "dissolved":
            irr_sink = ImplicitSourceTerm(
                coeff=-CellVariable(mesh=mesh, value=D_irr * scaling)
            )
            irr_source = CellVariable(mesh=mesh, value=D_irr * props["top"] * scaling)
        else:
            irr_sink = 0.0
            irr_source = 0.0

        # Assemble
        # Divided form uses coefficient 1.0 for TransientTerm
        eq = (
            TransientTerm(coeff=scaling) + conv_term
            == diff_term + lhs_term + rhs_term + irr_sink + irr_source
        )
        equations.append((var, eq))

    return equations


def build_steady_state_equations(
    mp, c, k, species_list, mesh, D_mol, D_bio, D_irr, bc_map, diagenetic_reactions
):
    """
    DEPRECATED: Steady state solver now builds equations inside the loop.
    This function remains for compatibility but returns None.
    """
    return None


def safe_ratio(
    num: np.ndarray,
    den: np.ndarray,
    fill: Union[float, int],
) -> np.ndarray:
    """
    Return ``num / den`` element‑wise while protecting against division‑by‑zero.

    Parameters
    ----------
    num : np.ndarray
        Numerator array (any shape that broadcasts with ``den``).
    den : np.ndarray
        Denominator array. Zeros are handled gracefully.
    fill : float or int, optional
        Value to place where ``den == 0``.  The default is ``np.nan``.
        Use ``0`` if you prefer a zero‑filled result.

    Returns
    -------
    np.ndarray
        Array of the same shape as the broadcasted inputs containing the
        element‑wise ratios. Positions where ``den`` is zero contain ``fill``.

    Notes
    -----
    * ``np.divide`` is used with the ``where`` argument – this avoids the
      creation of intermediate infinities and suppresses the runtime warning.
    * The output array is allocated with ``np.empty_like(num, dtype=float)`` so
      the result is always a floating‑point array, even if the inputs are integer.
    """
    # Ensure float output – division of ints would truncate otherwise
    out = np.empty_like(num, dtype=float)

    # Perform division only where denominator is non‑zero
    np.divide(num, den, out=out, where=den != 0)

    # Fill the “bad” positions
    out[den == 0] = fill
    return out


def calculate_k_iron_reduction(fes3, h2s):
    """
    Calculates the rate constant k_FeOx-SII for an array of ratios.
    Fes3+/H2S

    Based on Equation 46 and 47 from Halevy et al. (2023).
    """
    # 1. Define the piecewise conditions for half-life (tau_1/2) in hours
    # Condition 1: Ratio < 1 -> tau = 1.5h [cite: 1462]
    # Condition 2: 1 <= Ratio <= 2 -> Linear transition [cite: 1463]
    # Condition 3: Ratio > 2 -> tau = 0.5h [cite: 1464]

    ratios = safe_ratio(fes3, h2s, 10.0)

    tau_half = np.piecewise(
        ratios,
        [ratios < 1, (ratios >= 1) & (ratios <= 2), ratios > 2],
        [1.5, lambda r: 0.5 + 1.0 * (2.0 - r), 0.5],
    )

    # 2. Calculate the rate constant k = 0.693 / tau_1/2
    k_values = 0.693 / tau_half

    return k_values / (60 * 60 * 24 * 1e3)


def mol_to_weight_percent(c, mw, d):
    """Convert from mol/m^3 to weight percent.

    c: cocentration in mol/m^3
    mw: molar weight of substance
    d: density of sediment in gram/cm^3

    returns: wt% between 0 to 100
    """
    return 100 * c * mw / (d * 1e6)


def weight_percent_to_mol(wp, mw, d):
    """Convert from weight % to mol/m^3
    wp : weight percentage
    mw: mol weight
    d: density in gr/cm^3

    returns concentration in mol/m^3
    """

    return wp * d * 1e4 / mw
