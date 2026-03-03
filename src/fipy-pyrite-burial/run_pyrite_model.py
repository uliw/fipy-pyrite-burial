"""Define a specific modeling scenario."""

from petsc4py import PETSc

if not hasattr(PETSc.KSP.ConvergedReason, "CONVERGED_ATOL_NORMAL"):
    PETSc.KSP.ConvergedReason.CONVERGED_ATOL_NORMAL = (
        PETSc.KSP.ConvergedReason.CONVERGED_ATOL_NORMAL_EQUATIONS
    )
if not hasattr(PETSc.KSP.ConvergedReason, "CONVERGED_RTOL_NORMAL"):
    PETSc.KSP.ConvergedReason.CONVERGED_RTOL_NORMAL = (
        PETSc.KSP.ConvergedReason.CONVERGED_RTOL_NORMAL_EQUATIONS
    )

import fipy.tools.comms.dummyComm

if not hasattr(fipy.tools.comms.dummyComm.DummyComm, "petsc4py_comm"):
    fipy.tools.comms.dummyComm.DummyComm.petsc4py_comm = property(
        fget=lambda x: PETSc.COMM_SELF
    )

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
from live_plot_lib import LivePlotter

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

print("Starting run_pyrite_model.py...", flush=True)

experiment = "msr_h2s_ox_fe3_sorb_fe2_ox_fes"
# state_in = "statenpz.npz"
state_in = "statenpz.npz"
state_out = "statenpz2.npz"
# state_out = None

p_dict = {
    "bc_fe3": weight_percent_to_mol(0.0001, 56, 2.6),
    "bc_fe3": weight_percent_to_mol(1, 56, 2.6),
    "DB_depth": 0,
    "DB0": 4e-12 * 0,
    "max_steps": 100,  # max number of iterations
    "tolerance": 1e-12,  # convergence criterion
    "dt_tolerance": 1e-6,  # steady state threshold (stop simulation)
    "dt_target_change": 10.0,  # target change per step (for dt adaptation)
    "state_data": state_in,  # read state data
    # "plot_name": f"{experiment}.csv",
    "dt_init": Q_("1 hour").to("seconds").magnitude,  # initial dt
    "dt_max": Q_("1 day").to("seconds").magnitude,  # time step in years
    "dt_min": Q_("1 minute").to("seconds").magnitude,  # time step in years
    "t_end": Q_("10 kyr").to("seconds").magnitude,
    "solver": "non_steady",  # use non-steady solver, non_steady or steady
    # "solver": "bdf",  # use non-steady solver, non_steady or steady
    "solver_backend": "default",  # see solver_calls for options
    "initial_spacing": 0.01,  # meters
    "reaction_zone_spacing": 0.001,  # meters
    "max_spacing": 0.1,  # meters, None = no cap
    "reaction_zone": (1e-2, 2e-1),  # in meters
    "report_step": 10,  # how often to update plot
}

# add reactions as needed
p_dict["diagenetic_reactions"] = [
    rn.aerobic_respiration,
    rn.sulfate_reduction,
    rn.hs_oxidation,
    rn.elemental_sulfur_oxidation,
    rn.sulfide_mediated_iron_reduction,
    rn.fes_formation_only,
    rn.fes_dissolution,
    rn.fe2_oxidation,
    rn.fes_oxidation,
]

p_dict["instantenous_reactions"] = [
    rn.fe2_sorption_clip,
    rn.sulfide_speciation_clip,
    # rn.fes_precipitation_clip,
]
# initialize live plotter
plotter = LivePlotter(
    layout_path=p_dict.get("layout_file", "plot_layout.py"),
    display_length=p_dict.get("display_length", 2.0),
    output_path=p_dict.get("plot_name", "pyrite_model_fipy.csv").replace(".csv", ".pdf"),
)
plotter.start()


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
) = pyrite_model(p_dict, plot_queue=plotter.queue)

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

if state_out:
    save_state(c, state_out)

plotter.stop()
