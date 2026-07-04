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
    from pathlib import Path

    import pint
    import reactions_new as rn
    from pyrite_base_model import pyrite_model
    from reaction_constants_slow import get_reaction_constants

    from fipyrite.diff_lib import data_container, get_total_delta, save_data, save_state

    import faulthandler
    faulthandler.enable()

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    # set experiment name equal to script name
    experiment = Path(__file__).stem
    state_out = f"{experiment}_state.npz"

    p_dict = {
        "experiment": experiment,
        # "state_data": "run_pyrite_model_year_test_slow_isotopes_full_bak.npz",
        "process_monitor": "none",  # gui | video | none
        "layout_file": "plot_layout_velde.py",
        "layout_file": "plot_layout.py",
        # Solver Parameters
        "max_steps": 20000,  # max number of iterations
        "max_depth": 1,  # meters
        "t_end": Q_("10 kyr").to("seconds").magnitude,
        "dt_min": Q_("1 day").to("seconds").magnitude,  # time step in years
        "dt_init": Q_("1 week").to("seconds").magnitude,  # initial dt
        "dt_max": Q_("1 week").to("seconds").magnitude,  # time step in years
        "dt_target_change": 100,  # target change per step (for dt adaptation)
        "report_step": 100,  # how often to update plot
        "BT0": Q_("4 cm^2/year").to("m^2/second").magnitude,
        "BT_depth": Q_("7.6 cm").to("meter").magnitude,  # Bioturbation depth in m
        "BT_attenuation": Q_("2 cm").to("meter").magnitude,  # xbm of Velde et al.
        "om_o2_consumption": 1,  # Velde uses a 1:1 ratio
        "isotopes": True,
        # "bc_om_fast": Q_("1000 umol/(cm^2 * year)").to("mol/(m^2 * second)").magnitude,
        "reaction_constants": get_reaction_constants,  # see imports to select a different one
    }

    k = data_container()
    _k1, k = get_reaction_constants(7.5, 0.8, k_values=k)

    p_dict["diagenetic_reactions"] = [
        [rn.aerobic_respiration, k],
        [rn.dissimilatory_iron_reduction, k],
        [rn.sulfate_reduction, k],
        [rn.hs_oxidation, k],
        [rn.elemental_sulfur_oxidation, k],
        [rn.sulfide_mediated_iron_reduction_old, k],
        [rn.fe2_oxidation, k],
        [rn.fes_precipitation_terminal, k],
        [rn.fes_dissolution, k],
        # [rn.fes_precipitation_dissolution_linearized, k],
        [rn.fes_oxidation, k],
        [rn.pyrite_formation_s0, k],
        [rn.pyrite_formation_fes_ts2, k],
        [rn.pyrite_oxidation, k],
        [rn.s0_disproportionation, k],
    ]

    p_dict["instantenous_reactions"] = [
        # [rn.fes_equilibrium_clip, k],
        # [rn.fe2_sorption_clip, 1.0],
    ]

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
    df, fqfn = save_data(
        mp, c, k, species_list, z, D_mol, diagenetic_reactions, rn.equilibrium_reactions
    )

    if mp.isotopes:
        print(f"d34S = {get_total_delta(c, mp):.2f}")
        print(f"d34S = {get_total_delta(c, mp):.2f}")

    total_iron = df.c_fe2_total + df.c_fe3 + df.c_fes + df.c_fes2
    print(f"total_iron diff = {total_iron.max() - total_iron.min():.2e}")

    if state_out:
        save_state(c, state_out)
