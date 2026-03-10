"""Convert reaction constants.

Note, Velde et al likely have a typo in the FeS equilibrium constant
Rickard (2006) defines two reactionm constants, one with Ksp = 3.5
and the other with Ksp = - 3.5. Here we use the latter since otherwise
we would never precipitate FeS unless H2S > 1 mmol/l
"""


def get_reaction_constants(phi, pH):
    """convert reaction_constants into model units.
    Note: Velde et al report relative to bulk volume.
    so mass/(volume * time) needs to be multiplied by 1/phi
    and vice verso.
    """
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    velde: dict = {
        "poc_o2": [5e-10, "m^3/(mol*second)", "m^3/(mol*second)"],  # POC + O2 -> CO2
        "poc_so4": [1e-11, "m^3/(mol*second)", "m^3/(mol*second)"],  # POC + SO4 -> CO2
        "fe2_p_eq": [696, "dimensionless", "dimensionless"],  # sorbed vs Fe2+ Fe2+ liq.
        "fes2_ox": [
            1e-10,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
        ],  # FeS2 + O2 -> SO4, Halevy et al
        "fes_s0": [
            5e-8,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
        ],  # FeS + S0 -> FeS2, TBD ???
        "fes_ts2": [
            5e-8,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
        ],  # FeS + H2S -> FeS2, at 10C -> notes.org
        "hplus": [10 ** (-pH), "mol/l", "mol/m^3"],
        # Oxidation reactions after Velde at eal 2016
        "hs_ox": [1e7 * phi, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "fe3_hs": [
            494 * (1 - phi),
            "cm^3/(umol*year)",
            "m^3/(mol*second)",
        ],  # check with halevy
        "fes_ox": [1e7 * (1 - phi), "cm^3/(umol*year)", "m^3/(mol*second)"],
        "fe2_ox": [1e7 * phi, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "fe2p_ox": [1e7 * (1 - phi), "cm^3/(umol*year)", "m^3/(mol*second)"],
        "s0_ox": [4e7 * phi, "cm^3/(umol*year)", "m^3/(mol*second)"],
        # Iron sulfide reactions after Velde at eal 2016
        "fes_isp": [
            (1 - phi) * 1e4,
            "umol/((cm^3*year))",
            "mol/(m^3 *second)",
        ],  # FeS precipitation
        "fes_isd": [(1 - phi) * 3, "1/year", "1/second"],  # FeS dissolution
        "fes_sp": [  # FeS equilibrium constant
            (1 - phi) * 10**-3.5,
            "mol/l",
            "mol/m^3 ",
        ],  # FeS Saturation constant
    }

    k_values: dict = {}
    for k, v in velde.items():
        k_values[k] = Q_(f"{v[0]} {v[1]}").to(f"{v[2]}").magnitude

    return velde, k_values


if __name__ == "__main__":
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    velde, k_values = get_reaction_constants(0.65, 7.5)
    ureg = pint.UnitRegistry()
    for k, v in velde.items():
        n = Q_(f"{v[0]} {v[1]}").to(f"{v[2]}")
        print(f"{k} = {n.magnitude:.2e} [{n.units:~P}]")
