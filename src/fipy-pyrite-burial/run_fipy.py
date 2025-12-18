"""Model pyrite precipitation.

As a function of organic matter availability, including isotopes
Model units are meter/second, mmol/liter, and meter
"""

import pathlib as pl

import numpy as np
import pandas as pd
import pint

from fipy import Grid1D, CellVariable, LinearLUSolver

# from fipy.terms.vanLeerConvectionTerm import VanLeerConvectionTerm

import plot_data_new
from diff_lib import (
    data_container,
    diff_coeff,
    bioturbation_profile,
    relax_solution,
    get_l_mass,
    get_delta,
    save_data,
    run_non_steady_solver,
    run_steady_state_solver,
    build_non_steady_equations,
    build_steady_state_equations,
)
from reactions_new import diagenetic_reactions

# from reactions_new import diagenetic_reactions
# -----------------------------------------------------------------------------
# 1. PARAMETERS & CONFIGURATION
# -----------------------------------------------------------------------------
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Calculate Organic Matter BC
OM_wt = 4  # wt%
OM_Mol = OM_wt / 100 * 2600 / 12 * 1000  # C in mmol/l

mp = data_container({
    "plot_name": "pyrite_model_fipy.csv",
    "layout_file": "plot_layout.py",  # Plot layout file
    "max_depth": 4.0,  # meters
    "display_length": 2,  # meters
    "grid_points": 300,  # number of cells
    "temp": [0, 10.2],  # temp top, bottom, in C
    "phi": 0.65,  # porosity
    "w": Q_("46 cm/kyr").to("m/s").m,  # sedimentation rate in m/s
    "advection": 0,  # upward directed flow component
    "so4_d": 21,  # seaater deltae
    "VPDB": 0.044162589,  # VPDB reference ratio
    "msr_alpha": 1.055,  # MSR enrichment factor in mUr
    "bc_om": OM_Mol,  # mmol/l
    "bc_so4": 28.0,  # mmol/l
    "bc_s0": 0.0,  # mmol/l
    "bc_fe3": 60.0,  # mmol/l
    "bc_fe3": 600.0,  # mmol/l
    "DB0": 1e-8,  # Bioturbation coefficient
    "BI0": 0.001,  # Irrigation coefficient
    "DB_depth": 1.0,  # Bioturbation depth in m
    "BI_depth": 0.0,  # Irrigation depth (0 = off)
    "eps": 1e-4,  # limiters
    "relax": 0.8,  # relaxation parameter
    "tolerance": 1e-6,  # convergence criterion
    "dt_max": 100,  # time step in years
    "max_steps": 2000,  # max number of iterations
    "run_time": 3e4,  # run time in years
    "steady_state": True,  # assume steady state?
})

# Reaction Constants (k)
k = data_container({
    "poc_o2": 9e-10,  # POC + O2 -> CO2
    "poc_so4": 7e-13,  # POC + SO4 -> H2S
    "h2s_ox": 1e-10,  # H2S + O2 -> S0
    "fes_ox": 5e-12,  # FeS + O2 -> Fe3 + SO4
    "fes2_ox": 5e-12,  # FeS2 + O2 -> SO4
    "fes_s0": 1e-10,  # FeS + S0 -> FeS2
    "fes_h2s": 1e-11,  # FeS + H2S -> FeS2
    "fe3_h2s": 5e-11,  # Fe3 + H2S -> FeS * S0
})

mp.bc_so4_32 = get_l_mass(mp.bc_so4, mp.so4_d, mp.VPDB)

# -----------------------------------------------------------------------------
# 2. MESH GENERATION (Variable Grid)
# -----------------------------------------------------------------------------
nx = mp.grid_points
L = mp.max_depth

# Calculate geometric ratio to fit L with nx points, starting small
ratio = 1.005
dx_min = L * (ratio - 1) / (ratio**nx - 1)
dx_array = dx_min * ratio ** np.arange(nx)
mesh = Grid1D(dx=dx_array)

# Depth array (cell centers) for profiles
z = mesh.cellCenters[0].value

# -----------------------------------------------------------------------------
# 3. VARIABLES & DIFFUSION PROFILES
# -----------------------------------------------------------------------------
species_list = [
    "so4",
    "so4_32",
    "h2s",
    "h2s_32",
    "o2",
    "poc",
    "fe3",
    "fes",
    "fes_32",
    "s0",
    "s0_32",
    "fes2",
    "fes2_32",
]
c = data_container()

# Initialize CellVariables
for species_name in species_list:
    setattr(
        c,
        species_name,
        CellVariable(name=species_name, mesh=mesh, value=0.0, hasOld=True),
    )

# -- Temperature & Porosity Profiles --
T_profile = np.linspace(mp.temp[0], mp.temp[1], nx)
phi_profile = np.ones(nx) * mp.phi


D_mol = data_container()
D_mol.so4 = diff_coeff(T_profile, 4.88, 0.232, mp.phi)
D_mol.so4_32 = D_mol.so4
D_mol.h2s = diff_coeff(T_profile, 10.4, 0.273, mp.phi)
D_mol.h2s_32 = D_mol.h2s
D_mol.o2 = (
    (0.2604 + 0.006363 * ((T_profile + 273.15) / 1)) * 1e-9 / (1 - np.log(mp.phi**2))
)
zeros = np.zeros(nx)
for species_name in ["poc", "fe3", "fes", "fes_32", "s0", "s0_32", "fes2", "fes2_32"]:
    setattr(D_mol, species_name, zeros)


# -- Bioturbation Profile (Robust Sigmoid) --
D_mol.D_bio = bioturbation_profile(z, mp.DB0, mp.DB_depth)
D_mol.D_irr = bioturbation_profile(z, mp.BI0, mp.BI_depth)

# -----------------------------------------------------------------------------
# 4. BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------
bc_map = {
    "so4": {"top": mp.bc_so4, "type": "dissolved"},
    "so4_32": {"top": mp.bc_so4_32, "type": "dissolved"},
    "h2s": {"top": 0.0, "type": "dissolved"},
    "h2s_32": {"top": 0.0, "type": "dissolved"},
    "poc": {"top": mp.bc_om, "type": "particulate"},
    "o2": {"top": 0.2, "type": "dissolved"},
    "s0": {"top": mp.bc_s0, "type": "particulate"},
    "s0_32": {"top": mp.bc_s0, "type": "particulate"},
    "fe3": {"top": mp.bc_fe3, "type": "particulate"},
    "fes": {"top": 0.0, "type": "particulate"},
    "fes_32": {"top": 0.0, "type": "particulate"},
    "fes2": {"top": 0.0, "type": "particulate"},
    "fes2_32": {"top": 0.0, "type": "particulate"},
}

for species_name, props in bc_map.items():
    var = getattr(c, species_name)
    var.setValue(props["top"])
    var.constrain(props["top"], mesh.facesLeft)
    var.faceGrad.constrain(0.0, mesh.facesRight)

# build equation system and solve
if mp.steady_state:
    run_steady_state_solver(
        mp,
        None,
        c,
        species_list,
        k,
        diagenetic_reactions,
        mesh,
        D_mol,
        D_mol.D_bio,
        D_mol.D_irr,
        bc_map,
    )
else:
    equations = build_non_steady_equations(
        mp,
        c,
        k,
        species_list,
        mesh,
        D_mol,
        D_mol.D_bio,
        D_mol.D_irr,
        bc_map,
        diagenetic_reactions,
    )
    run_non_steady_solver(mp, equations, c, species_list)

# -----------------------------------------------------------------------------
# 8. EXPORT DATA
# -----------------------------------------------------------------------------
df, fqfn = save_data(mp, c, k, species_list, z, D_mol, diagenetic_reactions)

# 9. PLOTTING
# -----------------------------------------------------------------------------
# plt_desc = plot_data_new.load_layout_from_file(df, mp.layout_file)

# plot_data_new.plot(
#     df,
#     mp.display_length,
#     fqfn.with_suffix(".pdf"),
#     show=False,
#     plot_description=plt_desc,
#     measured_data_path="goldhaber_unified.csv",
# )
# print("Plot generated.")
