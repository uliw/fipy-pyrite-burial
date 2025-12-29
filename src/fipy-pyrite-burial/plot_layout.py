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
    from diff_lib import mol_to_weight_percent

    plt_desc = {
        "first_subplot": {
            "fig_width": 6,  # inches
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.c_so4, "SO4 [mmol/l]", {"color": "C0"}],
                # [df.c_s0, "S0 [mmol/l]", {"color": "C2"}],
                [df.c_h2s, "H2S [mmol/l]", {"color": "C1"}],
                [df.c_o2, "O2 [mmol/l]", {"color": "C3"}],
                [df.c_fe2, r"Fe$^{2+}$ [mmol]", {"color": "C8"}],
                # [df.c_fes2, r"FeS$_{2}$ [mmol]", {"color": "C7"}],
                # [df.c_fe3, r"Fe$_{3}^{+}$ [mmol]", {"color": "C5"}],
                # [df.c_fes, "FeS [mmol]", {"color": "C6"}],
            ],
            "yscale": "log",
            "xscale": "log",
            "ylim": (0.001, 1e2),
            "xlim": (0.01, None),
            "left_ylabel": r"[mmol/l]",
            "right": [
                # [df.c_poc, "OM [mmol]", {"color": "C4"}],
                [
                    mol_to_weight_percent(df.c_fes2, 32, 2.6),
                    r"FeS$_{2}$ [wt% S]",
                    {"color": "C7"},
                ],
                [
                    mol_to_weight_percent(df.c_fes, 32, 2.6),
                    "FeS [wt% S]",
                    {"color": "C6"},
                ],
                [
                    mol_to_weight_percent(df.c_s0, 32, 2.6),
                    "S0 [wt% S]",
                    {"color": "C2"},
                ],
            ],
            "right_ylim": (0.01, None),
            # "right_yscale": "log",
            "right_ylabel": "[wt %]S",
            "right1": [
                [
                    mol_to_weight_percent(df.c_fe3, 56, 2.6),
                    r"Fe$_{3}^{+}$ [wt% Fe]",
                    {"color": "C5"},
                ],
            ],
        },
        "second_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                # [df.f_o2, "f_o2", {"color": "C3"}],
                [df.f_so4, "f_so4", {"color": "C0"}],
                [df.f_h2s, "f_h2s", {"color": "C1"}],
                # [df.f_poc, "f_poc", {"color": "C4"}],
                [df.f_s0, "f_s0", {"color": "C2"}],
                [df.f_fe3, "f_fe3", {"color": "C5"}],
                [df.f_fes, "f_fes", {"color": "C6"}],
                [df.f_fes2, "f_fes2", {"color": "C7"}],
                [df.f_fe2, "f_fe2", {"color": "C8"}],
                # [df.D_bio, "D_bio", {"color": "C8"}],
            ],
            "xlim": (0.01, None),
            # "right": [df.D_irr, "D_irr", {"color": "C8"}],
            # "yscale": "symlog, linthresh=1e-14,linscale=0,1,base=10",
            "left_ylabel": "f [mol/m^3/s]",
            "xscale": "log",
            # "options-left": "set_yscale('symlog', linthresh=1e-14,linscale=0,1,base=10)",
        },
        "third_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.d_so4, "d_so4", {"color": "C0"}],
                [df.d_h2s, "d_h2s", {"color": "C1"}],
                [df.d_s0, "d_s0", {"color": "C2"}],
                [df.d_fes, "d_fes", {"color": "C6"}],
                [df.d_fes2, "d_fes2", {"color": "C7"}],
            ],
            # "right": [[df.d_h2s, "d_h2s", {"color": "C1"}]],
            # "yscale": "log",
            # "options-left": "set_ylim(1e-10, 1e-6)",
            "options-left": "set_ylim(-50, 100)",
            "xscale": "log",
            "xlim": (0.01, None),
        },
    }
    return plt_desc
