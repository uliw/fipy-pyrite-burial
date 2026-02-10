"""Model pyrite precipitation.

As a function of organic matter availability, including isotopes
Model units are meter/second, mmol/liter, and meter
"""


def run_model(p_dict: dict):
    """Model pyrite precipitation.

    As a function of organic matter availability, including isotopes
    Model units are meter/second, mmol/liter, and meter
    """
    import numpy as np
    import pint

    from fipy import CellVariable
    from diff_lib import (
        calculate_k_iron_reduction,
        data_container,
        diff_coeff,
        compute_sigmoidal_db,
        get_l_mass,
        # get_delta,
        weight_percent_to_mol,
        compute_bio_irrigation_alpha,
        make_grid,
        make_grid2,
        read_state,
        check_peclet_numbers,
    )
    from solver_calls import  run_non_steady_state_solver_coupled, run_steady_state_solver_coupled,

    from reactions_new import diagenetic_reactions
    from reaction_constants import get_reaction_constants

    # from reactions_old import diagenetic_reactions
    # from coupled_reactions import diagenetic_reactions

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    mp = data_container(
        {
            "plot_name": "pyrite_model_fipy.csv",
            "layout_file": "plot_layout.py",  # Plot layout file
            "steady_state": False,  # assume steady state?
            "max_depth": 10.0,  # meters
            "display_length": 2,  # meters
            "temp": [10.0, 10.1],  # temp top, bottom, in C
            "phi": 0.65,  # porosity
            "w": Q_("46 cm/kyr").to("m/s").m,  # sedimentation rate in m/s
            "advection": 0,  # upward directed flow component
            "pH": 7.5,  # porewater pH, Velde et al.
            "so4_d": 21,  # seawater delta
            "msr_alpha": 1.07,  # MSR enrichment factor in mUr
            "h2s_ox_alpha": 0.995,  # sulfide oxidation enrichment factor in mUr
            "bc_o2": 0.20,  # mmmol/l
            "bc_om": weight_percent_to_mol(4, 12, 2.6),  # wt% C
            "bc_so4": 28.0,  # mmol/l
            "bc_h2s": 0.0,  # mmol/l
            "bc_s0": 0.0,  # mmol/l
            "bc_fe2": 0,  # wt% Fe2
            "bc_fe2_p": 0,  # wt% sorbed Fe2
            "bc_fe3": weight_percent_to_mol(0.5, 56, 2.6),  # wt% Fe
            "DB0": 4e-12 * 0,  # Bioturbation coefficient
            "DB_depth": 0,  # Bioturbation depth in m
            "BI0": 1e-6 * 0,  # should be < 1e-5
            "BI_depth": 0.0,  # Irrigation depth (0 = off)
            "eps": 1e-8,  # limiters
            "relax": 0.1,  # use 0.1 for coupled solver, and 0.8 otherwise
            "tolerance": 1e-5,  # convergence criterion
            "dt_tolerance": 0.1,  # when to increase dt
            "dt_max": Q_("100 years").to("seconds").magnitude,  # time step in years
            "dt_min": Q_("1 second").to("seconds").magnitude,  # time step in years
            "max_steps": 2000,  # max number of iterations
            "t_end": Q_("10 kyear").to("seconds").magnitude,  # max model time
            "VCDT": 0.044162589,  # VCDT reference ratio
            "initial_spacing": 0.001,  # meters
            "reaction_zone_spacing": 0.001,  # meters
            reaction_zone: (5e-2, 2e-1),  # in meters
            "max_spacing": 0.01,  # meters, None = no cap
            "state_data": "state_data.npz",
        }
    )

    mp.update(p_dict)

    # Reaction Constants (k)
    k = {
        "poc_o2": 5e-10,  # POC + O2 -> CO2
        "poc_so4": 1e-12,  # POC + SO4 -> H2S # within range of Halevy 7e-12
        "fe2_p_eq": 696,  # partitioning of Fe2+ sorbed vs F2+ liquid , Velde at al.
        "fes2_ox": 1e-10,  # FeS2 + O2 -> SO4, Halevy et al
        "fes_s0": 5e-8,  # FeS + S0 -> FeS2, TBD ???
        "fes_h2s": 5e-8,  # FeS + H2S -> FeS2, at 10C -> notes.org
        # Fe3 + H2S -> FeS * S0 -> calculate_k_iron_reduction, Halevy
        # "fe3_h2s": calculate_k_iron_reduction(mp.bc_fe3, 0),  # ~1.6e-8
    }
    _k1, k2 = get_reaction_constants(mp.phi, mp.pH)
    k.update(k2)
    k = data_container(k)
    mp.bc_so4_32 = get_l_mass(mp.bc_so4, mp.so4_d, mp.VCDT)
    mp.bc_h2s_32 = get_l_mass(mp.bc_h2s, 0.0, mp.VCDT)  # Assume 0 delta for bc_h2s

    # -----------------------------------------------------------------------------
    # 2. MESH GENERATION (Variable Grid)
    # -----------------------------------------------------------------------------
    # mesh, z = make_grid(mp.max_depth, mp.initial_spacing, mp.max_spacing)
    mesh, z = make_grid2(mp.max_depth, mp.initial_spacing, mp.max_spacing)
    mp.grid_points = len(z)

    # -----------------------------------------------------------------------------
    # 3. VARIABLES & DIFFUSION PROFILES
    # -----------------------------------------------------------------------------
    species_list_full = [
        "so4",
        "so4_32",
        "h2s",
        "h2s_32",
        "o2",
        "poc",
        "fe2",
        "fe2_p",
        "fe2_total",
        "fe3",
        "fes",
        "fes_32",
        "s0",
        "s0_32",
        "fes2",
        "fes2_32",
        "hplus",
    ]

    # these are not part of the T & R equation system
    species_list_partial = species_list_full.copy()
    species_list_partial.remove("fe2")
    species_list_partial.remove("fe2_p")
    species_list_partial.remove("hplus")

    # ---- calculate some helper coefficients ----- #
    # Note: All of these assume that porosity does not change with time!

    # Porosity correction factor
    mp.fac_s = mp.phi / (1.0 - mp.phi)

    # Fe2 sorption fraction. Since sorption is faster than transport
    # we treat it as instantenous, i.e. it is just a function of
    # concentration
    # K_ads = k.fe2_p_eq  # 696

    # Check Units of K_ads!
    # If K_ads is dimensionless (Conc_solid_vol / Conc_liquid_vol):
    #   Capacity = phi + (1-phi)*K_ads
    # If K_ads is (Conc_solid_mass / Conc_liquid_vol) [L/kg]:
    #   Capacity = phi + (1-phi)*rho*K_ads
    R_factor = mp.phi + (1.0 - mp.phi) * k.fe2_p_eq

    # Calculate Fe2+ Fractions
    mp.f_diss = mp.phi / R_factor
    mp.f_sorb = (1.0 - mp.phi) * k.fe2_p_eq / R_factor

    # [H+] concentration. Move to c.hplus if using variable pH
    # mp.hplus = 10 ** (-mp.pH) * 1e6  # units are mol/m^3!

    # ---- Initialize CellVariables and diffusion coefficients ---- #
    D_mol = data_container()
    c = data_container()
    f = data_container()
    zeros = np.zeros(mp.grid_points)
    for species_name in species_list_full:
        setattr(D_mol, species_name, zeros)
        setattr(
            c,
            species_name,
            CellVariable(name=species_name, mesh=mesh, value=0.0, hasOld=True),
        )

    # -- Temperature & Porosity Profiles --
    T_profile = np.linspace(mp.temp[0], mp.temp[1], mp.grid_points)
    # phi_profile = np.ones(mp.grid_points) * mp.phi

    # ----- diffusion coefficients for liquid species ------ #
    D_mol.so4 = diff_coeff(T_profile, 4.88, 0.232, mp.phi)
    D_mol.so4_32 = D_mol.so4
    D_mol.h2s = diff_coeff(T_profile, 10.4, 0.273, mp.phi)
    D_mol.fe2 = diff_coeff(T_profile, 27.7, 1, mp.phi)
    D_mol.h2s_32 = D_mol.h2s
    D_mol.o2 = (
        (0.2604 + 0.006363 * ((T_profile + 273.15) / 1))
        * 1e-9
        / (1 - np.log(mp.phi**2))
    )

    # -- Bioturbation and Irrigation Profiles (Robust Sigmoid) --
    D_mol.D_irr = compute_bio_irrigation_alpha(z, mp.BI0, mp.BI_depth)
    D_mol.D_bio = compute_sigmoidal_db(z, mp.DB0, mp.DB_depth, 0.1)
    # lumped modeling of Fe2 liq and Fe2 adsorbed
    D_mol.fe2_total = 1 / (1 + k.fe2_p_eq) * D_mol.fe2

    # -----------------------------------------------------------------------------
    # 4. BOUNDARY CONDITIONS
    # -----------------------------------------------------------------------------
    bc_map = {
        "so4": {"top": mp.bc_so4, "type": "dissolved"},
        "so4_32": {"top": mp.bc_so4_32, "type": "dissolved"},
        "h2s": {"top": mp.bc_h2s, "type": "dissolved"},
        "h2s_32": {"top": mp.bc_h2s_32, "type": "dissolved"},
        "poc": {"top": mp.bc_om, "type": "particulate"},
        "o2": {"top": mp.bc_o2, "type": "dissolved"},
        "s0": {"top": mp.bc_s0, "type": "particulate"},
        "s0_32": {"top": mp.bc_s0, "type": "particulate"},
        "fe2_total": {"top": mp.bc_fe2, "type": "dissolved"},
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

    if mp.state_data:
        print(f"Reading state from {mp.state_data}")
        read_state(c, mp.state_data)

    check_peclet_numbers(mesh, mp, D_mol, species_list_partial, bc_map)

    # build equation system and solve
    if mp.steady_state:
        converged, step, total_time = run_steady_state_solver_coupled(
            mp,
            c,
            species_list_full,
            species_list_partial,
            k,
            diagenetic_reactions,
            mesh,
            D_mol,
            bc_map,
            z,
        )
    else:
        step, max_change = run_non_steady_state_solver_coupled(
            mp,
            c,
            species_list_full,
            species_list_partial,
            k,
            diagenetic_reactions,
            mesh,
            D_mol,
            bc_map,
            z,
        )
        converged = "Yes" if step < mp.max_steps else "No"
        total_time = 0.0
    return (
        mp,
        c,
        k,
        species_list_full,
        z,
        D_mol,
        diagenetic_reactions,
        converged,
        step,
        total_time,
    )


if __name__ == "__main__":
    import pint
    from diff_lib import (
        save_data,
        get_delta,
        weight_percent_to_mol,
        save_state,
        read_state,
    )

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
        "max_steps": 200,  # max number of iterations
        "tolerance": 1e-12,  # convergence criterion
        "dt_tolerance": 1e-4,  # convergence criterion
        # "state_data": state_in,  # read state data
        # "plot_name": f"{experiment}.csv",
        "dt_max": Q_("100 year").to("seconds").magnitude,  # time step in years
        "dt_min": Q_("100 year").to("seconds").magnitude,  # time step in years
        "t_end": Q_("30 kyear").to("seconds").magnitude,
        "steady_state": False,  # use non-steady solver
        "max_spacing": 0.01,  # meters, None = no cap
        "initial_spacing": 0.001,  # meters
    }
    # p_dict = {"bc_fe3": 1000, "DB_depth": 0.1, "max_depth": 10.0}

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
    ) = run_model(p_dict)

    # -----------------------------------------------------------------------------
    # 8. EXPORT DATA
    # -----------------------------------------------------------------------------
    df, fqfn = save_data(mp, c, k, species_list, z, D_mol, diagenetic_reactions)

    phi = mp.phi  # FIXME: do we need to scale this?
    s = phi * (df.c_so4.iloc[-1] + df.c_h2s.iloc[-1]) + (1 - phi) * (
        df.c_s0.iloc[-1] + df.c_fes.iloc[-1] + 2 * df.c_fes2.iloc[-1]
    )
    s32 = phi * (df.c_so4_32.iloc[-1] + df.c_h2s_32.iloc[-1]) + (1 - phi) * (
        df.c_s0_32.iloc[-1] + df.c_fes_32.iloc[-1] + df.c_fes2_32.iloc[-1]
    )

    d34s = get_delta(s, s32, mp.VCDT)
    print(f"d34S = {d34s:0.2f}, d34S pyrite = {df.d_fes2.iloc[-1]:.2f}")

    save_state(c, state_out)
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
