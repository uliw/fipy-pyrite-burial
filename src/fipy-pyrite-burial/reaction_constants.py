"""Define reaction constants.

Note, Velde et al likely have a typo in the FeS equilibrium constant
Rickard (2006) defines two reaction constants, one with Ksp = 3.5
and the other with Ksp = - 3.5. Here we use the latter since otherwise
we would never precipitate FeS unless H2S > 1 mmol/l
"""

from fipy import CellVariable


def get_reaction_constants(
    pH: float, phi: float | CellVariable, k_values: dict = {}
) -> dict:
    """Convert reaction_constants into model units.

    You need to specify if constants are specified in bulk or phase
    specific units. Note that liquid species have units of C/L and as
    such are already phase specific.
    Based on this information, the function will
    convert the units into the phase specific units required in the
    model. The exception of fe2_p_eq which is a ratio of solid-volume
    concentration and the porewater-volume concentration, and as such
    doe not need to be converted
    """
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    bulk, phase = True, False

    if isinstance(phi, CellVariable):
        phi = phi.value
        
    velde: dict = {
        # "poc_o2": [5e-11, "m^3/(mol*second)", "m^3/(mol*second)", bulk],  # POC + O2 -> CO2x
        # "poc_so4": [5e-11, "m^3/(mol*second)", "m^3/(mol*second)", phase],  # POC + SO4 -> CO2
        "poc_fast": [10, "1/year", "1/second", bulk],  # highly reactive OM
        "poc_slow": [0.1, "1/year", "1/second", bulk],  # reactive OM, Velde et al
        # "poc_slow": [
        #     1.2e-8,
        #     "1/second",
        #     "1/second",
        #     bulk,
        # ],  # reactive OM, Fossing et al.
        "fe2_p_eq": [
            696,
            "dimensionless",
            "dimensionless",
            phase,
        ],  # sorbed vs Fe2+ Fe2+ liq.
        "fes2_ox": [
            1e-10,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
            bulk,
        ],  # FeS2 + O2 -> SO4, Halevy et al
        "fes_s0": [
            5e-8,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
            bulk,
        ],  # FeS + S0 -> FeS2, TBD ???
        "fes_ts2": [
            5e-8,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
            bulk,
        ],  # FeS + H2S -> FeS2, at 10C -> notes.org
        "hplus": [10 ** (-pH), "mol/l", "mol/m^3", False],
        # Oxidation reactions after Velde at eal 2016
        "hs_ox": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)", bulk],
        "fe3_hs": [
            494,
            "cm^3/(umol*year)",
            "m^3/(mol*second)",
            bulk,
        ],  # check with halevy
        "fes_ox": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)", bulk],
        "fe2_ox": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)", phase],
        "fe2p_ox": [1e73, "cm^3/(umol*year)", "m^3/(mol*second)", bulk],
        "s0_ox": [4e7, "cm^3/(umol*year)", "m^3/(mol*second)", bulk],
        # Iron sulfide reactions after Velde at eal 2016
        "fes_isp": [
            1e4,
            "umol/((cm^3*year))",
            "mol/(m^3 *second)",
            bulk,
        ],  # FeS precipitation
        "fes_isd": [3, "1/year", "1/second", bulk],  # FeS dissolution
        "fes_sp": [  # FeS equilibrium constant
            10**-3.5,
            "mol/l",
            "mol/m^3 ",
            bulk,
        ],  # Dispro rate, 4 times that of HS oxidation
        "s0_dispro": [4e7, "cm^3/(umol*year)", "m^3/(mol*second)", bulk],
    }

    for k, v in velde.items():
        k_values[k] = Q_(f"{v[0]} {v[1]}").to(f"{v[2]}").magnitude
        if v[3]:
            k_values[k] = k_values[k] / (1 - phi)

    return velde, k_values


if __name__ == "__main__":
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    velde, k_values = get_reaction_constants(7.5, 0.8)
    ureg = pint.UnitRegistry()
    for k, v in velde.items():
        n = Q_(f"{v[0]} {v[1]}").to(f"{v[2]}")
        print(f"{k} = {n.magnitude:.2e} [{n.units:~P}]")

    print(f"k.fes_sp * k.hplus: {k_values['fes_sp'] * k_values['hplus']:.2e}")
