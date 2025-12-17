"""Model pyrite precipitation.

As a function of organic matter availability, including isotopes
Model units are meter/second, mmol/liter, and meter
"""

import pathlib as pl

import numpy as np
import pandas as pd
import pint

from fipy import Grid1D, CellVariable, LinearLUSolver

# Explicitly import other terms if needed, but DO NOT import SourceTerm from .terms
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm

from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
from fipy.terms.diffusionTerm import DiffusionTerm
from fipy.terms.transientTerm import TransientTerm
# from fipy.terms.vanLeerConvectionTerm import VanLeerConvectionTerm

import plot_data_new
from diff_lib import (
    data_container,
    diff_coeff,
    bioturbation_profile_2,
    relax_solution,
    get_l_mass,
    get_delta,
    run_solver,
    save_data,
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
    "max_depth": 4.0,  # meters
    "display_length": 4,  # meters
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
    # "bc_fe3": 600.0,  # mmol/l
    "DB_depth": 0.0,  # Bioturbation depth in m
    "DB0": 1e-8,  # Bioturbation coefficient
    "BI_depth": 0.0,  # Irrigation depth (0 = off)
    "BI0": 0.0001,  # Irrigation coefficient
    "eps": 1e-4,  # limiters
    "relax": 0.5,  # relaxation parameter
    "tolerance": 1e-7,  # convergence criterion
    "dt_max": 100,  # time step in years
    "max_steps": 2000,  # max number of iterations
    "run_time": 2e4,  # run time in years
})

# Reaction Constants (k)
k = data_container({
    "poc_o2": 3e-11,  # POC + O2 -> CO2
    "poc_so4": 3e-11,  # POC + SO4 -> H2S
    "h2s_ox": 1e-14,  # 1e-11 H2S + O2 -> S0
    "s0_fes": 4e-10,  # FeS + S0 -> FeS2
    "fes_h2s": 4e-10,  # Fes with H2S -> FeS2
    "fes_ox": 4e-14,  # 1e-11 FeS + O2 -> Fe3+ + SO4
    "fes2_s0": 1e-11,  # Pyrite Precipitation from S0
    "fes2_h2s": 1e-11,  # Pyrite Precipitation from H2S
    "fes2_ox": 1e-14,  # Pyrite Oxidation
    "fe3_h2s": 4e-10,  # Fe3 + H2S -> FeS * S0
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
D_bio = bioturbation_profile_2(z, mp.DB0, mp.DB_depth)
D_irr = bioturbation_profile_2(z, mp.BI0, mp.BI_depth)

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

# -----------------------------------------------------------------------------
# 6. BUILD EQUATIONS
# -----------------------------------------------------------------------------
equations = []
# Call reaction function once to set up the Terms
# We pass None for z, sc, results as they might not be used or needed for setup
f_res = diagenetic_reactions(mp, c, k, data_container())

for species_name in species_list:
    var = getattr(c, species_name)
    props = bc_map[species_name]
    # 1. Transport
    D_total = getattr(D_mol, species_name) + D_bio

    vel = mp.w
    if props["type"] == "dissolved":
        vel = mp.w - mp.advection

    # Explicitly create Rank 1 CellVariable for velocity to avoid shape errors
    u_var = CellVariable(mesh=mesh, value=([vel],), rank=1)

    # Use VanLeerConvectionTerm to minimize numerical dispersion artifacts in isotope ratios
    # conv_term = VanLeerConvectionTerm(coeff=u_var)
    conv_term = PowerLawConvectionTerm(coeff=u_var)

    # Wrap D_total in CellVariable to avoid shape ambiguity
    diff_term = DiffusionTerm(coeff=CellVariable(mesh=mesh, value=D_total))

    # 2. Reactions
    # Imported diagenetic_reactions returns (LHS_coeff, RHS_val, rate)
    lhs_val = getattr(f_res, species_name)[0]
    rhs_val = getattr(f_res, species_name)[1]

    # LHS from reactions_new is a constant or CellVariable expression.
    # We pass it directly to ImpicitSourceTerm to ensure it stays dynamic.
    lhs_term = ImplicitSourceTerm(coeff=lhs_val)

    # RHS from reactions_new is " - rate". We want "+ rate".
    # Use dynamic expression if it's a Variable, otherwise wrap static values
    if hasattr(rhs_val, "rank"):
        rhs_term = -rhs_val
    else:
        # It's an array or number (static)
        rhs_term = CellVariable(mesh=mesh, value=-rhs_val)

    # 3. Irrigation
    # Wrap D_irr to ensure Rank 0
    irr_sink = ImplicitSourceTerm(coeff=-CellVariable(mesh=mesh, value=D_irr))

    # FIX: Wrap in SourceTerm
    # irr_source_val is an array, wrapping in CellVariable is safer than SourceTerm
    irr_source = CellVariable(mesh=mesh, value=D_irr * props["top"])

    # Assemble
    eq = (
        TransientTerm(coeff=mp.phi) + conv_term
        == diff_term + lhs_term + rhs_term + irr_sink + irr_source
    )
    equations.append((var, eq))

# -----------------------------------------------------------------------------
# 7. SOLVE
# -----------------------------------------------------------------------------
run_solver(mp, equations, c, species_list)

# -----------------------------------------------------------------------------
# 8. EXPORT DATA
# -----------------------------------------------------------------------------
df, fqfn = save_data(mp, c, k, species_list, z, D_mol, D_bio, diagenetic_reactions)

# -----------------------------------------------------------------------------
# 9. PLOTTING
# -----------------------------------------------------------------------------
plt_desc = {
    "first_subplot": {
        "xaxis": [df.z, "Depth [m]"],
        "left": [
            [df.c_so4, "SO4 [mmol]", {"color": "C0"}],
            [df.c_s0, "S0 [mmol]", {"color": "C2"}],
        ],
        "right": [
            # [df.c_o2, "O2 [μmol]", {"color": "C3"}],
            # [df.c_poc, "OM [mmol]", {"color": "C4"}],
            [df.c_h2s, "H2S [mmol]", {"color": "C1"}],
            [df.c_fes, "FeS [mmol]", {"color": "C6"}],
            [df.c_fe3, r"Fe$_{3}^{+}$ [mmol]", {"color": "C5"}],
            [df.c_fes2, r"FeS$_{2}$ [mmol]", {"color": "C7"}],
        ],
        "left_ylabel": r"SO$_{4}$ & H$_{2}$S [mmol/l]",
        # "right_ylabel": "O2 [μmol/l]",
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
        ],
        "# yscale": "symlog",
        "left_ylabel": "f [mol/m^3/s]",
        # "options-left": "set_yscale('symlog', linthresh=1e-14,linscale=1e-14,base=10)",
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
        "options-left": "set_ylim(-40, 50)",
    },
}

plot_data_new.plot(
    df,
    mp.display_length,
    fqfn,
    # show=True,
    # isotopes=False,
    plot_description=plt_desc,
)
print("Plot generated.")
