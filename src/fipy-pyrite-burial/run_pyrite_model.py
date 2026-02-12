"""Define a specific modeling scenario."""

import pint
from diff_lib import (
    save_data,
    get_delta,
    weight_percent_to_mol,
    save_state,
    read_state,
)

from pyrite_base_model import pyrite_model

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

experiment = "msr_h2s_ox_fe3_sorb_fe2_ox_fes"
state_in = "msr_h2s_ox_fe3_sorb_fe2_ox.npz"
state_out = f"statenpz"

p_dict = {
    "bc_fe3": weight_percent_to_mol(0.0001, 56, 2.6),
    "bc_fe3": weight_percent_to_mol(1, 56, 2.6),
    "DB_depth": 0,
    "DB0": 4e-12 * 0,
    "relax": 0.8,  # use 0.1 with with coupled solver, and 0.8 with regular solver
    "max_steps": 100,  # max number of iterations
    "tolerance": 1e-9,  # convergence criterion
    "dt_tolerance": 1e-4,  # convergence criterion for time stepping
    # "state_data": state_in,  # read state data
    # "plot_name": f"{experiment}.csv",
    "dt_max": Q_("1000 year").to("seconds").magnitude,  # time step in years
    "dt_min": Q_("100 year").to("seconds").magnitude,  # time step in years
    "t_end": Q_("100 kyear").to("seconds").magnitude,
    "steady_state": False,  # use non-steady solver
    "initial_spacing": 0.01,  # meters
    "reaction_zone_spacing": 0.001,  # meters
    "max_spacing": 0.1,  # meters, None = no cap
    "reaction_zone": (1e-2, 2e-1),  # in meters
    "report_step": 10,  # how often to update plot
}

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
) = pyrite_model(p_dict)

# -----------------------------------------------------------------------------
# 8. EXPORT DATA
# -----------------------------------------------------------------------------
df, fqfn = save_data(mp, c, k, species_list, z, D_mol, diagenetic_reactions)

phi = mp.phi  # FIXME: do we need to scale this?
s = phi * (df.c_so4.iloc[-1] + df.c_ts2.iloc[-1]) + (1 - phi) * (
    df.c_s0.iloc[-1] + df.c_fes.iloc[-1] + 2 * df.c_fes2.iloc[-1]
)
s32 = phi * (df.c_so4_32.iloc[-1] + df.c_ts2_32.iloc[-1]) + (1 - phi) * (
    df.c_s0_32.iloc[-1] + df.c_fes_32.iloc[-1] + df.c_fes2_32.iloc[-1]
)

d34s = get_delta(s, s32, mp.VCDT)
print(f"d34S = {d34s:0.2f}, d34S pyrite = {df.d_fes2.iloc[-1]:.2f}")

save_state(c, state_out)
