"""
Configuration for plotting diagenetic modeling results.

This module defines the layout and aesthetic properties of the plots.
"""


def get_layout(df):
    """
    Return a dictionary describing the plot structure.

    Args:
        df: DataFrame containing the data to plot.

    Returns:
        dict: The plot description dictionary.
    """
    from fipyrite.diff_lib import (
        solid_conc_to_wt_percent,
        liquid_conc_to_wt_percent,
        solid_conc_to_umol_per_g,
    )

    lt = 2
    xlim = (0, 10)
    ylim = (None, None)

    if "d_SO4" in df.columns:
        isotopes = True
    else:
        isotopes = False

    # define color scheme
    poc_attr = {"color": "forestgreen", "lw": lt}
    O2_attr = {"color": "#AA4499", "lw": lt}
    SO4_attr = {"color": "cornflowerblue", "lw": lt, "fill_only": True, "zorder": -10}
    SO4_attr_nf = {"color": "cornflowerblue", "lw": lt, "fill_only": False}
    TS2_attr = {"color": "slateblue", "zorder": 10, "lw": lt}
    S0_attr = {"color": "#009988", "lw": lt}
    Fe3_attr = {"color": "saddlebrown", "lw": lt}
    Fe2_s_attr = {"color": "indianred", "lw": lt}
    Fe2_l_attr = {"color": "#cc3311", "lw": lt}
    FeS_attr = {"color": "#ee3377", "lw": lt}
    FeS2_attr = {"color": "darkorange", "lw": lt, "fill_only": True, "zorder": -9}
    FeS2_attr_nf = {"color": "darkorange", "lw": lt, "fill_only": False}
    dbio_attr = {"color": "#bbbbbb", "lw": lt, "fill_only": True}
    dirr_attr = {"color": "yellow", "lw": lt, "fill_only": True, "alpha": 0.1}

    phi = df.phi
    # relative to sulfur
    Fe3_s = solid_conc_to_wt_percent(df.c_Fe3, 32, 2.6, phi)
    FeS_s = solid_conc_to_wt_percent(df.c_FeS, 32, 2.6, phi)
    S0_s = solid_conc_to_wt_percent(df.c_S0, 32, 2.6, phi)
    FeS2_s = solid_conc_to_wt_percent(df.c_FeS2, 32, 2.6, phi)

    # relative to Iron
    Fe2_liquid = df.c_Fe2_total / 696
    Fe2_fe_sorbed = liquid_conc_to_wt_percent(df.c_Fe2_total, 56, 2.6, phi)
    Fe2 = liquid_conc_to_wt_percent(df.c_Fe2_total, 56, 2.6, phi)
    Fe3_fe = solid_conc_to_wt_percent(df.c_Fe3, 56, 2.6, phi)
    FeS_fe = solid_conc_to_wt_percent(df.c_FeS, 56, 2.6, phi)
    FeS2_fe = solid_conc_to_wt_percent(df.c_FeS2, 56, 2.6, phi)
    total_iron = Fe2_fe_sorbed + Fe3_fe + FeS_fe + FeS2_fe

    # umol/g
    Fe3_v = solid_conc_to_umol_per_g(df.c_Fe3, 2.6)
    FeS_v = solid_conc_to_umol_per_g(df.c_FeS, 2.6)
    Fe2_s_v = solid_conc_to_umol_per_g(df.c_Fe2_total, 2.6)

    plt_desc = {
        # -------------------------------- 1 panel ----------------------- #
        "first_subplot": {
            "xlim": xlim,
            "ylim": (0, 1),
            # "show_grid_options": {
            #     "grid": df.z,
            #     "step": 10,
            #     "color": "black",
            #     "alpha": 0.6,
            # },
            "fig_width": 6,  # inches
            "xaxis": [df.z * 100, "Depth [cm]"],
            # left axis
            "left": [
                # [df.c_SO4, r"SO$_{4}$", SO4_attr],
                [df.c_O2, r"O$_{2}$", O2_attr],
                [df.c_TS2 * 0.24, "TS$^{2-}$", TS2_attr],
                [Fe2_liquid, r"Fe$^{2+}_{liq}$", Fe2_l_attr],
            ],
            # "yscale": "log",
            # "xscale": "log",
            "left_ylabel": r"Concentration [mmol/L$_{PW}$]",
            # right 1
            "right1_ylim": (0, 150),
            "right1": [
                [Fe3_v, r"Fe$^{3+}$ [$\mu$mol/g]", Fe3_attr],
                [FeS_v, r"FeS [$\mu$mol/g]", FeS_attr],
                # [Fe2_s_v, r"Fe2 [$\mu$mol/g]", Fe2_s_attr],
                # [FeS_v, r"FeS [$\mu$mol/g]", FeS_attr],
            ],
            #    [FeS2_fe, r"FeS$_{2}$ [wt% Fe]", FeS2_attr],
            #    [FeS_fe * 1, r"FeS $\times$ 1 [wt% Fe]", FeS_attr],
            #    [Fe3_fe, r"Fe$^{3+}$ [wt% Fe]", Fe3_attr],
            #    [Fe2, r"Fe$^{2+}$ [wt% Fe]", Fe2_s_attr],
            # [
            #     total_iron,
            #     r"TFe [wt% Fe]",
            #     {"color": "black", "linestyle": "dotted"},
            # ],
            # ],
            # "right1_ylim": (0, 2),
            # "right1_ylabel": "[wt% Fe]",
            # # right 2
            # "right2": [
            #     [S0_s, r"S$^{0}$ [wt% S]", S0_attr],
            #     # [FeS2_s, r"FeS$_2$ [wt% S]", {"color": "black"}],
            #     # [FeS_s, r"$\times$ FeS [wt% S]", {"color": "C6"}],
            # ],
            # "right2_ylim": (0, 0.4),
            # "right2_ylabel": "[wt% S]",
        },
        # -------------------------------- 2 panel ----------------------- #
        "second_subplot": {
            "xaxis": [df.z * 100, "Depth [cm]"],
            "left": [
                [df.f_SO4, r"SO$_{4}$", SO4_attr_nf],
                [df.f_TS2, r"TS$^{2-}$", TS2_attr],
                [df.f_Fe2_total, r"Fe$^{2+}_{liq}$", Fe2_l_attr],
                [df.f_FeS, "FeS", FeS_attr],
                [df.f_Fe3, r"Fe${3+}$", Fe3_attr],
                [df.f_S0, r"S$^{0}$", S0_attr],
                # [df.f_O2, "O2", {"color": "C3"}],
                [df.f_poc_slow, r"POC", poc_attr],
                [df.f_poc_fast, r"POC", poc_attr],
                # [df.f_Fe2_p, "f_Fe2+p", {"color": "C9"}],
            ],
            "left_ylabel": r"reaction rate [mol m$^{-3}s^{-1}$]",
            # "xscale": "log",
            "options-left": "set_yscale('symlog', linthresh=1e-9,linscale=0.5,base=10)",
            "xlim": xlim,
            # "yscale": "symlog, linthresh=1e-14,linscale=0,1,base=10",
            # "right": [df.D_irr, "D_irr", {"color": "C8"}],
        },
        # -------------------------------- 3 panel ----------------------- #
    }
    if isotopes:
        plt_desc.update({
            "third_subplot": {
                "xaxis": [df.z * 100, "Depth [cm]"],
                "left": [
                    [df.d_SO4, r"SO$_{4}$", SO4_attr_nf],
                    [df.d_TS2, r"TS$^{2-}$", TS2_attr],
                    [df.d_FeS, r"FeS", FeS_attr],
                ],
                "left_ylabel": r"$\delta^{34}$ [mUr VCDT]",
                # "xscale": "log",
                "xlim": xlim,
                # "ylim": (-15, 75),
                # "right": [[df.d_h2s, "d_h2s", {"color": "C1"}]],
                # "yscale": "log",
                "options-left": "set_ylim(-30, 75)",
            },
        })

    return plt_desc
