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
    import matplotlib as mpl
    from fipyrite.diff_lib import solid_conc_to_wt_percent, liquid_conc_to_wt_percent
    # define color scheme
    lt = 2
    poc_slow_attr = {"color": "forestgreen", "lw": lt}
    poc_fast_attr = {"color": "lightgreen", "lw": lt}
    o2_attr = {"color": "#AA4499", "lw": lt}
    so4_attr = {"color": "cornflowerblue", "lw": lt, "fill_only": True, "zorder": -10}
    so4_attr_nf = {"color": "cornflowerblue", "lw": lt, "fill_only": False}
    ts2_attr = {"color": "slateblue", "zorder": 10, "lw": lt}
    s0_attr = {"color": "#009988", "lw": lt}
    fe3_attr = {"color": "saddlebrown", "lw": lt}
    fe2_s_attr = {"color": "indianred", "lw": lt}
    fe2_l_attr = {"color": "#cc3311", "lw": lt}
    fes_attr = {"color": "#ee3377", "lw": lt}
    fes2_attr = {"color": "darkorange", "lw": lt, "fill_only": True, "zorder": -9}
    fes2_attr_nf = {"color": "darkorange", "lw": lt, "fill_only": False}
    dbio_attr = {"color": "#bbbbbb", "lw": lt, "fill_only": True}
    dirr_attr = {"color": "yellow", "lw": lt, "fill_only": True, "alpha": 0.1}

    phi = df.phi
    # relative to sulfur
    fe3_s = solid_conc_to_wt_percent(df.c_fe3, 32, 2.6, phi)
    fes_s = solid_conc_to_wt_percent(df.c_fes, 32, 2.6, phi)
    s0_s = solid_conc_to_wt_percent(df.c_s0, 32, 2.6, phi)
    fes2_s = solid_conc_to_wt_percent(df.c_fes2, 32, 2.6, phi)

    # relative to Iron
    fe2_liquid = df.c_fe2_total / 696
    fe2_fe_sorbed = liquid_conc_to_wt_percent(df.c_fe2_total, 56, 2.6, phi)
    fe3_fe = solid_conc_to_wt_percent(df.c_fe3, 56, 2.6, phi)
    fes_fe = solid_conc_to_wt_percent(df.c_fes, 56, 2.6, phi)
    fes2_fe = solid_conc_to_wt_percent(df.c_fes2, 56, 2.6, phi)
    total_iron = fe2_fe_sorbed + fe3_fe + fes_fe + fes2_fe

    plt_desc = {
        "first_subplot": {
            # "show_grid_options": {
            #     "grid": df.z,
            #     "step": 10,
            #     "color": "black",
            #     "alpha": 0.6,
            # },
            "fig_width": 6,  # inches
            "xaxis": [df.z, "Depth [m]"],
            # left axis
            "left": [
                # [df.c_so4, r"SO$_{4}$", so4_attr],
                # [df.c_o2, r"O$_{2}$", o2_attr],
                [df.c_ts2 * phi, "TS$^{2-}$", ts2_attr],
                [df.c_fe2_total * phi, r"Fe2_total", fe2_s_attr],
                #[fe2_liquid, r"Fe$^{2+}_{liq}$", fe2_l_attr],
                #[df.c_poc_fast, "OM_f", poc_fast_attr],
                #[df.c_poc_slow, "OM_s", poc_slow_attr],
            ],
            "yscale": "log",
            "xscale": "log",
            "xlim": (1e-4, 1),
            "left_ylabel": r"Concentration [mmol/L$_{bulk}$]",
            # right 1
            "right1": [
                [df.c_fes * (1-phi), r"FeS", fes_attr],
                # [fes_fe * 1, r"FeS $\times$ 1 [wt% Fe]", fes_attr],
                # [fe3_fe, r"Fe$^{3+}$ [wt% Fe]", fe3_attr],
                # [fe2_fe_sorbed, r"Fe$^{2+}_{sorb}$ [wt% Fe]", fe2_s_attr],
                # [
                #     total_iron,
                #     r"TFe [wt% Fe]",
                #     {"color": "black", "linestyle": "dotted"},
                # ],
            ],
            "right1_ylabel": r"Concentration [mmol/L$_{bulk}$]",
            # right 2
            # "right2": [
            #     [s0_s, r"S$^{0}$ [wt% S]", s0_attr],
            #     # [fes2_s, r"FeS$_2$ [wt% S]", {"color": "black"}],
            #     # [fes_s, r"$\times$ FeS [wt% S]", {"color": "C6"}],
            # ],
            # "right2_ylim": (0, 0.4),
            # "right2_ylabel": "[wt% S]",
        },
        "second_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                # [df.f_so4, r"SO$_{4}$", so4_attr_nf],
                [df.f_ts2, r"TS$^{2-}$", ts2_attr],
                [df.f_fe2_total, r"Fe$^{2+}_{liq}$", fe2_l_attr],
                [df.f_fes, "FeS", fes_attr],
                [df.f_fe3, r"Fe${3+}$", fe3_attr],
                # [df.f_fes2, r"FeS$_{2}$", fes2_attr_nf],
                #  [df.f_s0, r"S$^{0}$", s0_attr],
                [df.D_bio * 1e6, r"1e6 $\times$ D$_{bio}$ [$m^{2}/s$]", dbio_attr],
                # [df.D_irr, "1e6 $\times$ D$_{irr}$ [1/s]", dirr_attr],
                # [df.f_o2, "O2", o2_attr],
                # [df.f_poc_slow, r"OM_s", poc_slow_attr],
                # [df.f_poc_fast + df.f_poc_fast, r"OM_f", poc_fast_attr],
                # [df.f_fe2_p, "f_fe2+p", {"color": "C9"}],
            ],
            "left_ylabel": r"reaction rate [mol m$^{-3}s^{-1}$]",
            "xscale": "log",
            "options-left": "set_yscale('symlog', linthresh=1e-9,linscale=0.5,base=10)",
            "xlim": (1e-4, 2),
            # "yscale": "symlog, linthresh=1e-14,linscale=0,1,base=10",
            # "right": [df.D_irr, "D_irr", {"color": "C8"}],
        },
        "third_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.d_so4, r"SO$_{4}$", so4_attr_nf],
                [df.d_ts2, r"TS$^{2-}$", ts2_attr],
                [df.d_fes, r"FeS", fes_attr],
                [df.d_fes2, r"FeS$_{2}$", fes2_attr_nf],
                # [df.d_s0, r"S$^{0}$", s0_attr],
            ],
            "left_ylabel": r"$\delta^{34}$ [mUr VCDT]",
            "xscale": "log",
            "xlim": (1e-4, 2),
            # "ylim": (-15, 75),
            # "right": [[df.d_h2s, "d_h2s", {"color": "C1"}]],
            # "yscale": "log",
            "options-left": "set_ylim(-30, 75)",
        },
    }
    return plt_desc
