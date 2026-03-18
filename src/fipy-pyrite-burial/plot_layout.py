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
    from diff_lib import solid_conc_to_wt_percent, liquid_conc_to_wt_percent

    phi = df.phi
    fe2_s = liquid_conc_to_wt_percent(df.c_fe2_total, 32, 2.6, phi)
    fe3_s = solid_conc_to_wt_percent(df.c_fe3, 32, 2.6, phi)
    fes_s = solid_conc_to_wt_percent(df.c_fes, 32, 2.6, phi)
    s0_s = solid_conc_to_wt_percent(df.c_fes, 32, 2.6, phi)

    fes2_s = solid_conc_to_wt_percent(df.c_fes2, 32, 2.6, phi)
    fe2_fe = liquid_conc_to_wt_percent(df.c_fe2_total, 56, 2.6, phi)
    fe3_fe = solid_conc_to_wt_percent(df.c_fe3, 56, 2.6, phi)
    fes_fe = solid_conc_to_wt_percent(df.c_fes, 56, 2.6, phi)
    fes2_fe = solid_conc_to_wt_percent(df.c_fes2, 56, 2.6, phi)
    total_iron = fe2_fe + fe3_fe + fes_fe + fes2_fe

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
            "left": [
                [df.c_so4, "SO4", {"color": "C0"}],
                [df.c_ts2, "TS$^{2-}$", {"color": "C1"}],
                [df.c_o2, "O2", {"color": "C3"}],
                [df.c_fe2, r"Fe$^{2+}_{liq}$", {"color": "C8"}],
            ],
            "yscale": "log",
            "xscale": "log",
            "ylim": (1e-6, 1e5),
            "xlim": (1e-2, None),
            "left_ylabel": r"[mmol/l]",
            "right": [
                [
                    fes2_s,
                    r"FeS$_2$ [wt% S]",
                    {"color": "black"},
                ],
                [
                    fes_s * 10,
                    r"10 $\times$ FeS [wt% S]",
                    {"color": "C6"},
                ],
                [
                    s0_s,
                    "S0 [wt% S]",
                    {"color": "C2"},
                ],
            ],
            "right_ylim": (0, 4),
            "right_ylabel": "[wt% S]",
            "right2": [
                [
                    fe3_fe,
                    r"Fe$^{3+}$ [wt% Fe]",
                    {"color": "C5"},
                ],
            ],
            "right2_ylim": (0, 1),
            "right2_ylabel": "[wt% Fe]",
        },
        "second_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                # [df.f_o2, "f_o2", {"color": "C3"}],
                [df.f_so4, "f_so4", {"color": "C0"}],
                [df.f_ts2, r"TS$^{2-}$", {"color": "C1"}],
                # [df.f_poc, "f_poc", {"color": "C4"}],
                [df.f_s0, "f_s0", {"color": "C2"}],
                [df.f_fe3, "f_fe3", {"color": "C5"}],
                [df.f_fes2, "f_fes2", {"color": "C7"}],
                [df.f_fe2_total, "f_fe2_total", {"color": "C8"}],
                [df.f_fes, "f_fes", {"color": "C6"}],
                # [df.f_fe2_p, "f_fe2+p", {"color": "C9"}],
                # # [df.D_bio, "D_bio", {"color": "C8"}],
            ],
            "xlim": (1e-2, None),
            # "right": [df.D_irr, "D_irr", {"color": "C8"}],
            # "yscale": "symlog, linthresh=1e-14,linscale=0,1,base=10",
            "left_ylabel": r"f [mol m$^{-3}s^{-1}$]",
            "xscale": "log",
            "options-left": "set_yscale('symlog', linthresh=1e-14,linscale=0.5,base=10)",
        },
        "third_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.d_so4, "d_so4", {"color": "C0"}],
                [df.d_ts2, r"d TS$^{2-}$", {"color": "C1"}],
                [df.d_s0, "d_s0", {"color": "C2"}],
                [df.d_fes, "d_fes", {"color": "C6"}],
                [df.d_fes2, "d_fes2", {"color": "C7"}],
            ],
            # "right": [[df.d_h2s, "d_h2s", {"color": "C1"}]],
            # "yscale": "log",
            # "options-left": "set_ylim(1e-10, 1e-6)",
            "options-left": "set_ylim(-50, 100)",
            "xscale": "log",
            "xlim": (1e-2, None),
        },
    }
    return plt_desc
