"""Define a specific modeling scenario."""

import pint
from diff_lib import (
    save_data,
    get_delta,
    weight_percent_to_mol,
    save_state,
    # read_state,
)

from pyrite_base_model import pyrite_model
import reactions_new as rn

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# print statement moved to __main__ block

experiment = "pyrite_model_fipy"
# state_in = "statenpz.npz"
state_in = None
state_out = "state.npz"
# state_out = None

p_dict = {
    # "bc_fe3": weight_percent_to_mol(0.0001, 56, 2.6),
    "t_end": Q_("1 kyear").to("seconds").magnitude,
    "max_steps": 30,  # max number of iterations
    "dt_min": Q_("1 minute").to("seconds").magnitude,  # time step in years
    "dt_init": Q_("1 hour").to("seconds").magnitude,  # initial dt
    "dt_max": Q_("1 year").to("seconds").magnitude,  # time step in years
    "process_monitor": "gui",  # gui | video | none
    "plot_name": f"{experiment}",
    "isotopes": False,
    "report_step": 10,  # how often to update plot
    "bc_fe3": weight_percent_to_mol(1, 56, 2.6),
    "DB_depth": 0,
    "DB0": 4e-12 * 0,
    "tolerance": 1e-12,  # convergence criterion
    "dt_tolerance": 1e-6,  # steady state threshold (stop simulation)
    "dt_target_change": 1.0,  # target change per step (for dt adaptation)
    "state_data": state_in,  # read state data
    "solver": "non_steady",  # use non-steady solver, non_steady or steady
    # "solver_backend": "LinearLUSolver",  # see solver_calls for options
    "solver_backend": "LinearGMRESSolver",  # see solver_calls for options
    "solver_backend": "default",  # see solver_calls for options
    "initial_spacing": 0.01,  # meters
    "reaction_zone_spacing": 0.001,  # meters
    "max_spacing": 0.1,  # meters, None = no cap
    "reaction_zone": (1e-2, 2e-1),  # in meters
}

# add reactions as needed
p_dict["diagenetic_reactions"] = [
    rn.aerobic_respiration,
    rn.sulfate_reduction,
    rn.hs_oxidation,
    rn.elemental_sulfur_oxidation,
    rn.sulfide_mediated_iron_reduction,
    rn.fe2_oxidation,
    # rn.fes_unified_reaction,
    # rn.fes_oxidation,
    # rn.pyrite_formation_s0,
    # rn.pyrite_formation_fes_ts2,
    # rn.pyrite_oxidation,
]

p_dict["instantenous_reactions"] = [
    rn.fe2_sorption_clip,
    rn.sulfide_speciation_clip,
]
if __name__ == "__main__":
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
    ) = pyrite_model(p_dict)

    # -----------------------------------------------------------------------------
    # 8. EXPORT DATA
    # -----------------------------------------------------------------------------
    df, fqfn = save_data(mp, c, k, species_list, z, D_mol, diagenetic_reactions)

    print(f"d34S = {rn.get_total_delta(c, mp):.2f}")
    from diff_lib import mol_to_weight_percent

    total_iron = df.c_fe2_total + df.c_fe3 + df.c_fes + df.c_fes2
    print(f"total_iron diff = {total_iron.max() - total_iron.min():.2e}")

    if state_out:
        save_state(c, state_out)

# LivePlotter is now managed inside pyrite_model
