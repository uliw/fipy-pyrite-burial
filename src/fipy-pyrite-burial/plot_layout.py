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
    # define color scheme
    o2_attr =  {"color" : "forestgreen", "linestyle" : "dashed"}
    
    so4_attr = {"color" : "cornflowerblue", "linestyle" : "dashed"}
    ts2_attr = {"color" : "slateblue",  "linestyle" : "dashed", "zorder": 10}
    s0_attr = {"color" : "cornflowerblue"} 
    
    fe3_attr = {"color": "saddlebrown"}
    fe2_s_attr = {"color": "indianred"}
    fe2_l_attr = {"color": "brown",  "linestyle" : "dashed"}
    fes_attr = {"color" : "red"}
    fes2_attr = {"color" : "darkorange"}

    phi = df.phi
    # relative to sulfur
    fe3_s = solid_conc_to_wt_percent(df.c_fe3, 32, 2.6, phi)
    fes_s = solid_conc_to_wt_percent(df.c_fes, 32, 2.6, phi)
    s0_s = solid_conc_to_wt_percent(df.c_s0, 32, 2.6, phi)
    fes2_s = solid_conc_to_wt_percent(df.c_fes2, 32, 2.6, phi)

    # relative to Iron
    fe2_liquid = df.c_fe2_total/696
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
                [df.c_o2, r"O$_{2}$", o2_attr],
                [df.c_so4, r"SO$_{4}$", so4_attr],
                [df.c_ts2, "TS$^{2-}$",ts2_attr],
                [fe2_liquid, r"Fe$^{2+}_{liq}$", fe2_l_attr],
            ],
            "yscale": "log",
            "xscale": "log",
            "ylim": (1e-6, 1e5),
            "xlim": (1e-2, None),
            "left_ylabel": r"[mmol/l]",
            # right 1
            "right1": [
                [fe2_fe_sorbed, r"Fe$^{2+}_{sorb}$ [wt% Fe]", fe2_s_attr],
                [fe3_fe, r"Fe$^{3+}$ [wt% Fe]", fe3_attr],
                [fes_fe, r"FeS [wt% Fe]", fes_attr],
                [fes2_fe, r"FeS$_{2}$ [wt% Fe]", fes2_attr],
                # [
                #     total_iron,
                #     r"TFe [wt% Fe]",
                #     {"color": "black", "linestyle": "dotted"},
                # ],
            ],
            "right1_ylim": (0, 10),
            "right1_ylabel": "[wt% Fe]",
            # right 2
            "right2": [
                # [fes2_s, r"FeS$_2$ [wt% S]", {"color": "black"}],
                # [fes_s, r"$\times$ FeS [wt% S]", {"color": "C6"}],
                [s0_s, r"S$^{0}$ [wt% S]", s0_attr],
            ],
            "right2_ylim": (0, 4),
            "right2_ylabel": "[wt% S]",
            
        },
        "second_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                # [df.f_o2, "f_o2", {"color": "C3"}],
                [df.f_so4, r"f_SO$_{4}$", so4_attr],
                [df.f_ts2, r"f_TS$^{2-}$", ts2_attr],
                #[df.f_poc, r"f_O2", {"color": "C4"}],
                [df.f_fe2_total, r"f_Fe$^{2+}_{sorb}$",fe2_l_attr],
                [df.f_fe3, r"f_Fe${3+}$", fe3_attr],
                [df.f_fes, "f_FES", fes_attr],
                [df.f_fes2, r"f_FeS$_{2}$", fes2_attr],
                [df.f_s0, r"f_S$^{0}$", s0_attr],
                # [df.f_fe2_p, "f_fe2+p", {"color": "C9"}],
                # # [df.D_bio, "D_bio", {"color": "C8"}],
            ],
            # "right": [df.D_irr, "D_irr", {"color": "C8"}],
            # "yscale": "symlog, linthresh=1e-14,linscale=0,1,base=10",
            "left_ylabel": r"f [mol m$^{-3}s^{-1}$]",
            "xscale": "log",
            "options-left": "set_yscale('symlog', linthresh=1e-10,linscale=0.5,base=10)",
            "xlim": (1e-2, None),
        },
        "third_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.d_so4, r"$\delta^{34}$S SO$_{4}$",  so4_attr],
                [df.d_ts2, r"$\delta^{34}$S TS$^{2-}$", ts2_attr],
                [df.d_s0, r"$\delta^{34}$S S$^{0}$", s0_attr],
                [df.d_fes, r"$\delta^{34}$S FeS", fes_attr],
                [df.d_fes2, r"$\delta^{34}$S FeS$_{2}$", fes2_attr],
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
