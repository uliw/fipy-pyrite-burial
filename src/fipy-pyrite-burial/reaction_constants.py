"""Convert reaction constants."""


def get_reaction_constants():
    """convert reaction_constants into model units."""
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    velde: dict = {
        # Oxidation reactions after Velde at eal 2016
        "h2s_ox": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "fe3_h2s": [494, "cm^3/(umol*year)", "m^3/(mol*second)"],  # check with halevy
        "fes_ox": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "fe2_ox": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        # Iron sulfide reactions after Velde at eal 2016
        "isp_fes": [
            1e4,
            "umol/((cm^3*year))",
            "mol/(m^3 *second)",
        ],  # FeS precipitation
        "isp_isd": [3, "1/year", "1/second"],  # FeS dissolution
    }

    k_values: dict = {}
    for k, v in velde.items():
        k_values[k] = Q_(f"{v[0]} {v[1]}").to(f"{v[2]}").magnitude

    return velde, k_values


if __name__ == "__main__":
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    velde, k_values = get_reaction_constants()
    ureg = pint.UnitRegistry()
    for k, v in velde.items():
        n = Q_(f"{v[0]} {v[1]}").to(f"{v[2]}")
        print(f"{k} = {n.magnitude:.2e} [{n.units:~P}]")
