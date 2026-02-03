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
from concurrent.futures import ThreadPoolExecutor

_executor = None


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
    import numpy

    with np.errstate(divide="ignore", invalid="ignore"):
        h = c - li
        d = np.where(li < 0.001, float("nan"), 1000 * (h / li - r) / r)

    return d


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
    """Calculate the light isotope mass from the total mass and delta.

    :param m: total mass/concentration
    :param d: delta
    :param r: reference ratio

    :return: light isotope mass
    """
    return (1000.0 * m) / ((d + 1000.0) * r + 1000.0)


def get_total_s_export(df, VCDT=0.044162589):
    """Calculate the total sulfur and delta34S at the bottom of the column.
    Uses the logic as defined in run_fipy.py.
    """
    import numpy as np

    phi = df.phi.iloc[-1]
    s = phi * (df.c_so4.iloc[-1] + df.c_h2s.iloc[-1]) + (1 - phi) * (
        df.c_s0.iloc[-1] + df.c_fes.iloc[-1] + 2 * df.c_fes2.iloc[-1]
    )
    s32 = phi * (df.c_so4_32.iloc[-1] + df.c_h2s_32.iloc[-1]) + (1 - phi) * (
        df.c_s0_32.iloc[-1] + df.c_fes_32.iloc[-1] + df.c_fes2_32.iloc[-1]
    )
    h = s - s32
    d34s = np.where(s32 < 0.001, np.nan, 1000 * (h / s32 - VCDT) / VCDT)
    return float(s), float(d34s)


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


def compute_sigmoidal_db(z, Db0, xL, xbm):
    """
    Computes the bio-diffusivity (Db) at a specific depth (z)
    using Equation 4 from van de Velde and Meysman (2016).

    Parameters:
    z   : float or np.ndarray
          Depth into the sediment in m
    Db0 : float
          Bio-diffusivity coefficient m^2/s
    xL  : float
          Depth of the mixed layer in m
    xbm : float
          Attenuation coefficient determining the width of the transition zone [m]

    Returns:
    float or np.ndarray: The bio-diffusivity at depth z.
    """
    z_cm = z * 100
    xL_cm = xL * 100
    xbm_cm = xbm * 100

    # Define the term used in the exponents
    exponent_term = -(z_cm - xL_cm) / (0.25 * xbm_cm)

    # Use a robust sigmoid formula to avoid overflow/NaN warnings
    # We clip the exponent to a range that won't overflow double precision floats (~700)
    # This still results in 0.0 or Db0 as expected at the limits.
    Db_z = Db0 / (1.0 + np.exp(np.clip(-exponent_term, -700, 700)))

    return Db_z


def compute_bio_irrigation_alpha(z, alpha0, x_irr):
    """
    Computes the bio-irrigation coefficient (alpha) at a specific depth (z)
    using Equation 6 from van de Velde and Meysman (2016).

    Parameters:
    z     : float or np.ndarray
            Depth into the sediment (m).
    alpha0: float
            Bio-irrigation coefficient at the sediment-water interface(SWI).
    x_irr : float
            Attenuation coefficient determining the depth of irrigation (m).

    Returns:
    float or np.ndarray: The irrigation intensity at depth z.
    """
    z = z * 100
    x_irr = x_irr * 100
    # Equation 6: alpha(z) = alpha0 * exp(-z / x_irr)
    alpha_z = alpha0 * np.exp(-z / x_irr)

    return alpha_z


def make_grid(L, initial_spacing, max_spacing, r=1.05):
    """
    Construct a 1D grid driven by spacing constraints.

    Growth is geometric (at rate r) until max_spacing is reached.

    Args:
    -----
        L (float): Total length of the domain (max depth).
        initial_spacing (float): The size of the first cell (at z=0).
        max_spacing (float): Maximum cell size allowed.
        r (float): Geometric growth rate (default 1.05).

    Returns:
    -------
        tuple (mesh, z_centers)
            mesh: A fipy.Grid1D object.
            z_centers: A numpy array of cell center coordinates.
    """
    from fipy import Grid1D
    import numpy as np

    if initial_spacing >= max_spacing:
        initial_spacing = max_spacing

    dx_list = []
    current_z = 0
    current_dx = initial_spacing

    # 1. Geometric Section
    while current_z + current_dx < L and current_dx < max_spacing:
        dx_list.append(current_dx)
        current_z += current_dx
        current_dx *= r

    # 2. Uniform Section
    if current_z < L:
        remaining_L = L - current_z
        n_uniform = int(np.ceil(remaining_L / max_spacing))
        if n_uniform > 0:
            actual_uniform_dx = remaining_L / n_uniform
            dx_list.extend([actual_uniform_dx] * n_uniform)

    dx_array = np.array(dx_list)
    N = len(dx_array)
    print(f"Grid generated with {N} points.")

    mesh = Grid1D(dx=dx_array)
    z_centers = mesh.cellCenters[0].value

    return mesh, z_centers


def save_data(mp, c, k, species_list, z, D_mol, diagenetic_reactions):
    """
    Save the model results to a CSV file (Synchronous).
    """
    f_final = diagenetic_reactions(mp, c, k, data_container())
    return _save_data_to_disk(mp, c, k, species_list, z, D_mol, f_final)


def _save_data_to_disk(mp, c, k, species_list, z, D_mol, f_final):
    """Internal helper to write data to disk."""
    import pandas as pd
    import pathlib as pl

    data = {"z": z}

    def get_v(obj):
        """Helper to extract value from FiPy variables or return as-is."""
        if hasattr(obj, "value"):
            return obj.value
        return obj

    for species_name in species_list:
        data[f"c_{species_name}"] = get_v(getattr(c, species_name))

    for species_name in species_list:
        res_tuple = getattr(f_final, species_name)
        rates_val = res_tuple[2]  # Index 2 is the RATES
        data[f"f_{species_name}"] = get_v(rates_val)

    # Save all items in D_mol
    for d_name, d_val in D_mol.items():
        data[d_name] = get_v(d_val)

    # calculate delta values
    for species_name in species_list:
        if "_32" in species_name:
            base_species = species_name[:-3]
            if base_species == "fes2":
                s = 2 * data[f"c_{base_species}"]
            else:
                s = data[f"c_{base_species}"]
            s32 = data[f"c_{species_name}"]
            data[f"d_{base_species}"] = get_delta(s, s32, mp.VCDT)

    data["w"] = np.ones(len(z)) * mp.w
    data["phi"] = np.ones(len(z)) * mp.phi

    df = pd.DataFrame(data)
    fqfn = pl.Path.cwd() / mp.plot_name
    df.to_csv(fqfn, index=False)
    # print(f"Data saved to {fqfn}")

    return df, fqfn


def save_state(c, filename):
    """
    Save the current concentration profiles (state) to a compressed numpy file.

    :param c: data_container containing species as CellVariables
    :param filename: Path to save the NPZ file
    """
    from pathlib import Path

    path = Path(filename)
    if path.exists():
        path.rename(f"{filename}-old")

    state_dict = {}
    for key, var in c.items():
        if hasattr(var, "value"):
            state_dict[key] = var.value

    np.savez_compressed(filename, **state_dict)
    print(f"State saved to {filename}")


def read_state(c, filename):
    """
    Read concentration profiles (state) from a compressed numpy file and update c.

    :param c: data_container containing species as CellVariables
    :param filename: Path to the NPZ file
    """
    import pathlib as pl

    if not pl.Path(filename).exists():
        print(f"Warning: State file {filename} not found. Skipping initialization.")
        return False

    with np.load(filename) as data:
        for key in data.files:
            if hasattr(c, key):
                var = getattr(c, key)
                if hasattr(var, "setValue"):
                    var.setValue(data[key])
                else:
                    c[key] = data[key]
            else:
                print(
                    f"Warning: Species {key} found in state file but not in model container."
                )

    print(f"State loaded from {filename}")
    return True


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


def calculate_k_iron_reduction(fe3, h2s):
    """
    Calculates the rate constant k_FeOx-SII for an array of ratios.
    Fe3+/H2S

    Based on Equation 46 and 47 from Halevy et al. (2023).
    """
    # 1. Define the piecewise conditions for half-life (tau_1/2) in hours
    # Condition 1: Ratio < 1 -> tau = 1.5h [cite: 1462]
    # Condition 2: 1 <= Ratio <= 2 -> Linear transition [cite: 1463]
    # Condition 3: Ratio > 2 -> tau = 0.5h [cite: 1464]

    ratios = safe_ratio(fe3, h2s, 10.0)

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


def _get_executor():
    """Create a module‑wide ThreadPoolExecutor on first use."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1)
    return _executor


def save_data_async(mp, c, k, species_list, z, D_mol, diagenetic_reactions) -> None:
    """
    Schedule a background write of model results to CSV and return immediately.
    """
    # 1. Capture current rates in the main thread (FiPy objects aren't thread-safe)
    f_final = diagenetic_reactions(mp, c, k, data_container())

    # 2. Snapshot values (numpy arrays) to avoid race conditions during solver update
    def snap(obj):
        if hasattr(obj, "value"):
            return obj.value.copy()
        if hasattr(obj, "copy"):
            return obj.copy()
        return obj

    c_snap = data_container({s: snap(getattr(c, s)) for s in species_list})
    f_snap = data_container(
        {s: (None, None, snap(getattr(f_final, s)[2])) for s in species_list}
    )
    D_mol_snap = data_container({key: snap(val) for key, val in D_mol.items()})

    # Copy metadata
    mp_snap = data_container(mp)
    k_snap = data_container(k)
    z_snap = z.copy()

    # 3. Submit to background worker
    _get_executor().submit(
        _save_data_to_disk,
        mp_snap,
        c_snap,
        k_snap,
        species_list,
        z_snap,
        D_mol_snap,
        f_snap,
    )

    return None, None


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
    f_init = diagenetic_reactions(mp, c, k, f=data_container())

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
        f_res = diagenetic_reactions(mp, c, k, f=data_container())

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

        if step % 10 == 0 or step == 1:
            # if step > 0:
            print(
                f"Iteration {step}: Max Var Change {max_change:.2e}, Coupled Residual {res:.2e}"
            )
            df, fqfn = save_data_async(
                mp, c, k, species_list_full, z, D_mol, diagenetic_reactions
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

    start_wall = time.time()
    print(
        f"Starting Pseudo-Transient Solver (dt={mp.dt_max / (60 * 60 * 24 * 365):.1e} yr)"
    )

    solver = LinearLUSolver(tolerance=mp.tolerance)

    # 1. PRE-BUILD THE EQUATIONS
    # --------------------------
    lhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}
    rhs_vars = {s: CellVariable(mesh=mesh, value=0.0) for s in species_list_partial}
    cross_vars = {s: {} for s in species_list_partial}

    # Analyze Topology
    f_init = diagenetic_reactions(mp, c, k, f=data_container())

    for s in species_list_partial:
        res = getattr(f_init, s)
        if len(res) > 3:
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
        conv_term = PowerLawConvectionTerm(coeff=u_var, var=var)
        diff_term = DiffusionTerm(coeff=CellVariable(mesh=mesh, value=D_total), var=var)

        # -- Reactions --
        lhs_term = ImplicitSourceTerm(coeff=lhs_vars[species_name], var=var)
        rhs_term = rhs_vars[species_name]

        cross_reaction_terms = 0.0
        for source_name, cv in cross_vars[species_name].items():
            source_var = getattr(c, source_name)
            cross_reaction_terms += ImplicitSourceTerm(coeff=cv, var=source_var)

        # -- Irrigation --
        if props["type"] == "dissolved":
            irr_sink = ImplicitSourceTerm(
                coeff=CellVariable(mesh=mesh, value=-D_mol.D_irr), var=var
            )
            irr_source = CellVariable(mesh=mesh, value=D_mol.D_irr * props["top"])
        else:
            irr_sink = 0.0
            irr_source = 0.0

        # Assemble Equation
        eq = (
            TransientTerm(var=var) + conv_term
            == diff_term + lhs_term + rhs_term + cross_reaction_terms + irr_sink + irr_source
        )
        eqs.append(eq)

    # Create the Coupled Equation System
    coupled_eq = reduce(lambda a, b: a & b, eqs)

    max_change = 1e10
    step = 0
    total_time = 0

    # 2. Time Stepping Loop
    while max_change > mp.tolerance and step < mp.max_steps:
        step += 1

        # A. Update Old Values (Advance Time)
        for s in species_list_partial:
            getattr(c, s).updateOld()

        # Snapshot for convergence check
        last_sol = {s: getattr(c, s).value.copy() for s in species_list_partial}

        # B. Update Reaction Rates (Non-Linear Step)
        f_res = diagenetic_reactions(mp, c, k, f=data_container())

        # update placeholders
        for species_name in species_list_partial:
            res_tuple = getattr(f_res, species_name)
            lhs_val = res_tuple[0]
            rhs_val = res_tuple[1]

            # Update Diagonal
            lhs_vars[species_name].setValue(getattr(lhs_val, "value", lhs_val))
            rhs_vars[species_name].setValue(getattr(rhs_val, "value", rhs_val))

            # Update Cross-Terms if present
            if len(res_tuple) > 3:
                batch_coeffs = {src: 0.0 for src in cross_vars[species_name]}
                for source_name, coeff in res_tuple[3]:
                    if source_name in batch_coeffs:
                        val = getattr(coeff, "value", coeff)
                        batch_coeffs[source_name] += val

                for src_name, total_coeff in batch_coeffs.items():
                    target_cv = cross_vars[species_name][src_name]
                    target_cv.setValue(total_coeff)

        # C. Sweep the Coupled System
        res = coupled_eq.sweep(dt=mp.dt_max, solver=solver)

        # D. Convergence Check
        total_time = total_time + mp.dt_max
        max_change = 0.0
        for s in species_list_partial:
            var = getattr(c, s)
            curr_val = var.value
            prev_val = last_sol[s]

            # Calculate absolute change
            diff = np.abs(curr_val - prev_val)
            max_change = max(max_change, np.max(diff))

        # Reporting
        if step % 10 == 0 or step == 1:
            years = total_time / (60 * 60 * 24 * 365)
            print(
                f"Time: {years:.2e} [yr], Step: {step}: Max Change: {max_change:.2e}, Residual {res:.2e}"
            )
            df, fqfn = save_data_async(
                mp, c, k, species_list_full, z, D_mol, diagenetic_reactions
            )

    # 3. Final Report
    status = "Converged" if step < mp.max_steps else "Failed"
    print(f"{status} in {step} steps. Wall time: {time.time() - start_wall:.2f}s")

    return step, max_change
