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
        # 1. Numerical Safeguards
        # Ensure total mass 'c' is at least 'li' (light isotope) to avoid negative heavy mass
        # and prevent division by very near zero or zero.
        li_safe = np.maximum(li, 1e-30)
        c_safe = np.maximum(c, li_safe)

        h = c_safe - li_safe
        ratio = h / li_safe

        # 2. Thresholding for NaN
        # If total concentration is effectively zero, delta is undefined (NaN)
        d = np.where(c_safe < 1e-6, np.nan, 1000 * (ratio - r) / r)

        # 3. Clipping Extreme Values
        # Delta values below -999 or above extreme limits are usually numerical artifacts at trace levels.
        # -1000 is the mathematical limit for 100% light isotope (0% heavy),
        # so anything significantly below -1000 is impossible.
        d = np.clip(d, -1000.0, 1000.0)

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
    f_final, RATES = diagenetic_reactions(mp, c, k, data_container())
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

    # 1. Collect all basic concentrations and rates
    for species_name in species_list:
        data[f"c_{species_name}"] = get_v(getattr(c, species_name))
        res_tuple = getattr(f_final, species_name)
        data[f"f_{species_name}"] = get_v(res_tuple[2])

    # 2. Save all items in D_mol (diffusion coefficients)
    for d_name, d_val in D_mol.items():
        data[d_name] = get_v(d_val)

    # 3. Calculate delta values for specific sulfur species
    # We prioritize common names (so4, h2s, fes, s0, fes2)
    isotope_map = {
        "so4": "so4_32",
        "h2s": "h2s_32",
        "hs": "hs_32",
        "ts2": "ts2_32",
        "fes": "fes_32",
        "s0": "s0_32",
        "fes2": "fes2_32",
    }

    for base, iso in isotope_map.items():
        if f"c_{base}" in data and f"c_{iso}" in data:
            s_total = data[f"c_{base}"]
            if base == "fes2":
                s_total = 2.0 * s_total
            s32 = data[f"c_{iso}"]
            data[f"d_{base}"] = get_delta(s_total, s32, mp.VCDT)

    data["w"] = np.ones(len(z)) * mp.w
    data["phi"] = np.ones(len(z)) * mp.phi

    df = pd.DataFrame(data)
    # Ensure no NaN from misalignment by strictly using the same structure
    fqfn = pl.Path.cwd() / mp.plot_name
    df.to_csv(fqfn, index=False)

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


def check_peclet_numbers(mesh, mp, D_mol, species_list, bc_map):
    # Get cell sizes (dx)
    # For a 1D mesh, mesh.dx is an array of cell lengths
    dx = mesh.dx

    for species in species_list:
        props = bc_map[species]

        # 1. Determine Velocity
        vel = mp.w
        if props["type"] == "dissolved":
            vel = mp.w - mp.advection

            # 2. Determine Diffusion
            # For solids, D_mol is usually 0, so D is just D_bio
            d_val = getattr(D_mol, species) + D_mol.D_bio

            # 3. Calculate Pe for every cell
            # We use a small epsilon to avoid division by zero
            pe_cells = (np.abs(vel) * dx) / (d_val + 1e-20)

            max_pe = np.max(pe_cells)

            if max_pe > 2:
                print(f"Species: {species:10} | Max Pe: {max_pe:.2f}")
                print(
                    f"  --> WARNING: Pe > 2. Numerical dispersion/oscillations likely."
                )


def save_data_async(
    mp, c, k, species_list, z, D_mol, diagenetic_reactions, current_dt
) -> None:
    """
    Schedule a background write of model results to CSV and return immediately.
    """
    from reactions_new import equilibrium_reactions

    # 1. Capture current rates in the main thread (FiPy objects aren't thread-safe)
    f_final, RATES = diagenetic_reactions(mp, c, k, data_container())
    f_final, RATES = equilibrium_reactions(mp, c, k, f_final, RATES, current_dt)

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


def get_time_units(t):
    """Adjust time units between second to ky."""
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    seconds = 60
    minutes = seconds * 60
    hours = minutes * 60
    days = hours * 24
    weeks = days * 7
    months = weeks * 4.5
    years = months * 12
    kyears = years * 1000
    Myears = kyears * 1000

    if t < seconds:
        t = Q_(f"{t} seconds")
    elif t < minutes:
        t = Q_(f"{t} seconds").to("minutes")
    elif t < hours:
        t = Q_(f"{t} seconds").to("hours")
    elif t < days:
        t = Q_(f"{t} seconds").to("days")
    elif t < weeks:
        t = Q_(f"{t} seconds").to("week")
    elif t < months:
        t = Q_(f"{t} seconds").to("month")
    elif t < years:
        t = Q_(f"{t} seconds").to("year")
    elif t < kyears:
        t = Q_(f"{t} seconds").to("kyear")
    else:
        t = Q_(f"{t} seconds").to("kyear")

    return t


def make_grid2(
    L: float,
    initial_spacing: float,
    reaction_zone_spacing: float,
    max_spacing: float,
    reaction_zone: tuple,
    r=1.05,
):
    """Build a variable distance mesh.

    The mesh will decrease from initial_spacing to reaction_zone_spacing, and then increases to max_spacing below the reaction zone. The reaction zone depth interval is given by the reaction_zone tuple e.g. (0.5, 1)

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

    rz_start, rz_end = reaction_zone
    dx_list = []
    current_z = 0

    # 1. Refinement phase: from surface to reaction zone
    current_dx = initial_spacing
    while current_z + current_dx < rz_start:
        dx_list.append(current_dx)
        current_z += current_dx
        current_dx = max(current_dx / r, reaction_zone_spacing)

    if current_z < rz_start:
        dx_list.append(rz_start - current_z)
        current_z = rz_start

    # 2. Reaction zone phase: uniform fine spacing
    n_rz = int(np.ceil((rz_end - rz_start) / reaction_zone_spacing))
    actual_rz_dx = (rz_end - rz_start) / n_rz
    dx_list.extend([actual_rz_dx] * n_rz)
    current_z = rz_end

    # 3. Coarsening phase: from reaction zone to L
    current_dx = actual_rz_dx * r
    while current_z + current_dx < L and current_dx < max_spacing:
        dx_list.append(current_dx)
        current_z += current_dx
        current_dx *= r

    # 4. Uniform Section
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
