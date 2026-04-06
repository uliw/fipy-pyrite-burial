"""
Define a specific modeling scenario.

Note: Model units are meter/second, concentrations are given mmol/liter (mol/m^3) and
solids are expressed as concentration per unit of solid volume (mmol/L_solid).

This keeps the physics of the "solid phase" independent of how much water is currently
squeezing around it.  If the sediment compacts (porosity ϕ decreases), the amount of
organic matter per gram of rock doesn't change, but the amount of organic matter per
liter of bulk sediment does.  ​
"""

if __name__ == "__main__":
    import pint
    from diff_lib import (
        save_data,
        # get_delta,
        # wt_percent_to_solid_conc,
        save_state,
        get_total_delta,
        # read_state,
    )

    from pyrite_base_model import pyrite_model
    import reactions_new as rn

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    experiment = "pyrite"
    state_out = f"{experiment}_state.npz"

    p_dict = {
        "experiment": experiment,
        "state_data": None,
        "process_monitor": "video",  # gui | video | none
    }
    # p_dict = {
    #     "max_steps": 140,  # max number of iterations
    #     "max_depth": 4.0,  # meters
    #     "t_end": Q_("10 kyr").to("seconds").magnitude,
    #     "dt_min": Q_("1 minute").to("seconds").magnitude,  # time step in years
    #     "dt_init": Q_("1 month").to("seconds").magnitude,  # initial dt
    #     "dt_max": Q_("1 year").to("seconds").magnitude,  # time step in years
    #     "process_monitor": "none",  # gui | video | none
    #     "process_monitor": "gui",  # gui | video | none
    #     "process_monitor": "video",  # gui | video | none
    #     "plot_name": f"{experiment}",
    #     "isotopes": True,
    #     "report_step": 2,  # how often to update plot
    #     "w": Q_("0.2 cm/yr").to("m/s").m,  # sedimentation rate in m/s
    #     "bc_o2": 6,  # mmmol/l
    #     "bc_fe3": Q_("12 umol/(cm^2 * year)")
    #     .to("mol/(m^2 * second)")
    #     .magnitude,  # mol C / (m²·s)
    #     "bc_om": Q_("548 umol/(cm^2 * year)").to("mol/(m^2 * second)").magnitude,
    #     "phi": 0.8,
    #     "DB_depth": 0.1,
    #     "DB0": Q_("4 cm^2/year").to("m^2/second").magnitude,
    #     "tolerance": 1e-12,  # convergence criterion
    #     "dt_tolerance": 1e-12,  # steady state threshold (stop simulation)
    #     "dt_target_change": 10,  # target change per step (for dt adaptation)
    #     "state_data": state_in,  # read state data
    #     "solver": "non_steady",  # use non-steady solver, non_steady or steady
    #     # "solver_backend": "LinearLUSolver",  # see solver_calls for options
    #     "solver_backend": "LinearGMRESSolver",  # see solver_calls for options
    #     "solver_backend": "default",  # see solver_calls for options
    #     "initial_spacing": 0.01,  # meters
    #     "reaction_zone_spacing": 0.001,  # meters
    #     "max_spacing": 0.1,  # meters, None = no cap
    #     "reaction_zone": (0.05, 0.8),  # in meters
    # }

    print("Starting run_pyrite_model.py...", flush=True)
    (
        mp,
        c,
        k,
        species_list,
        z,
        D_mol,
        diagenetic_reactions,
        converged,
        step,
        total_time,
    ) = pyrite_model(p_dict, experiment=experiment)

    # -----------------------------------------------------------------------------
    # 8. EXPORT DATA
    # -----------------------------------------------------------------------------
    df, fqfn = save_data(mp, c, k, species_list, z, D_mol, diagenetic_reactions)

    print(f"d34S = {get_total_delta(c, mp):.2f}")
    print(f"d34S = {get_total_delta(c, mp):.2f}")

    total_iron = df.c_fe2_total + df.c_fe3 + df.c_fes + df.c_fes2
    print(f"total_iron diff = {total_iron.max() - total_iron.min():.2e}")

    if state_out:
        save_state(c, state_out)

# LivePlotter is now managed inside pyrite_model
