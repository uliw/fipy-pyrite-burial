"""Define reaction constants.

Note, Velde et al likely have a typo in the FeS equilibrium constant.
Rickard (2006) defines two reaction constants, one with Ksp = 3.5
and the other with Ksp = - 3.5. Here we use the latter since otherwise
we would never precipitate FeS unless H2S > 1 mmol/l
"""

from fipy import CellVariable


def get_reaction_constants(
    pH: float, phi: float | CellVariable, k_values: dict = {}
) -> dict:
    """Convert reaction_constants into model units.

        It is assumed that the reaction constants are in phase specific units.
        This can be tricky, as Velde et al, state that their k-values are in
        bulk units, but then they convert it before using (see Table 3). E.g.
        they write:

       (1−ϕ)⋅kSIR⋅[FeOOH]⋅[HS−]
    ​
        which suggests that the (1−ϕ) is used to convert FeOOH from solid
        phase concentration to bulk concentration. Since this code
        tracks all concentrations in phase specific units, converting the
        k-value by deviding by (1−ϕ), would cancel out, and we can use the
        k-value as published.
    """
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    if isinstance(phi, CellVariable):
        phi = phi.value

    velde: dict = {
        "poc_fast": [10, "1/year", "1/second"],  # highly reactive OM
        "poc_slow": [0.1, "1/year", "1/second"],  # reactive OM, Velde et al
        "fe2_p_eq": [
            696,
            "dimensionless",
            "dimensionless",
        ],  # sorbed vs Fe2+ Fe2+ liq.
        "fes2_ox": [
            1e-10,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
        ],  # FeS2 + O2 -> SO4, Halevy et al
        "fes_s0": [  # FeS + S0 -> FeS2
            5e-8,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
        ],  # FeS + HS -> FeS2
        "fes_ts2": [
            5e-8,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
        ],  # FeS + H2S -> FeS2, at 10C -> notes.org
        "hplus": [10 ** (-pH), "mol/l", "mol/m^3"],
        # Oxidation reactions after Velde at eal 2016
        "hs_ox": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "fe3_hs": [
            494,
            "cm^3/(umol*year)",
            "m^3/(mol*second)",
        ],  # check with halevy
        "fes_ox": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "fe2_ox": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        # "fe2p_ox": [1e73, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "s0_ox": [4e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        # Iron sulfide reactions after Velde at eal 2016
        "fes_isp": [
            1e4,
            "umol/((cm^3*year))",
            "mol/(m^3 *second)",
        ],  # FeS precipitation
        "fes_isd": [3, "1/year", "1/second"],  # FeS dissolution
        "fes_sp": [  # FeS equilibrium constant
            10**-3.5,
            "mol/l",
            "mol/m^3 ",
        ],  # Dispro rate, 4 times that of HS oxidation
        "s0_dispro": [4e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
    }

    for k, v in velde.items():
        k_values[k] = Q_(f"{v[0]} {v[1]}").to(f"{v[2]}").magnitude

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
    print(f"k.hplus: {k_values['hplus']:.2e}")
