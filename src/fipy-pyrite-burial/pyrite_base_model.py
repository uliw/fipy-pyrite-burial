"""
Define a reaction-transport model that computes pyrite precipitation.

as a function of organic matter availability, including isotopes. Model units are
meter/second, concentrations are given mmol/liter (mol/m^3) and solids are expressed as
concentration per unit of solid volume (mmol/L_solid).

This keeps the physics of the "solid phase" independent of how much water is currently
squeezing around it.  If the sediment compacts (porosity ϕ decreases), the amount of
organic matter per gram of rock doesn't change, but the amount of organic matter per
liter of bulk sediment does.  ​

As such, a reaction between a liquid and a solid needs to be scaled

f = k * [SO4] * (1 - phi)/phi * [OM]
"""


def pyrite_model(p_dict: dict, plot_queue=None):
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
        wt_percent_to_solid_conc,
        compute_bio_irrigation_alpha,
        make_grid,
        make_grid2,
        read_state,
        check_peclet_numbers,
    )
    from solver_calls import (
        run_non_steady_state_solver_coupled,
    )

    from reactions_new import diagenetic_reactions
    from reaction_constants import get_reaction_constants
    from live_plot_lib import LivePlotter

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    print("Entering pyrite_model...", flush=True)

    mp = data_container(
        {
            "plot_name": "pyrite_model_fipy",
            "layout_file": "plot_layout.py",  # Plot layout file
            "process_monitor": "video",  # gui | video | none
            "steady_state": False,  # assume steady state?
            "max_depth": 10.0,  # meters
            "display_length": 2,  # meters
            "temp": [10.0, 10.1],  # temp top, bottom, in C
            "phi": 0.65,  # porosity
            "w": Q_("46 cm/kyr").to("m/s").m,  # sedimentation rate in m/s
            "advection": 0,  # upward directed flow component
            "pH": 7.5,  # porewater pH, Velde et al.
            "isotopes": False,  # include isotope calculations
            "so4_d": 21,  # seawater delta
            "msr_alpha": 1.07,  # MSR enrichment factor in mUr
            "hs_ox_alpha": 0.995,  # sulfide oxidation enrichment factor in mUr
            "s0_ox_alpha": 1,  # sulfide oxidation enrichment factor in mUr
            "bc_o2": 0.20,  # mmmol/l
            "bc_om": wt_percent_to_solid_conc(4, 12, 2.6, 0.65),  # wt% C
            "bc_so4": 28.0,  # mmol/l
            "bc_ts2": 0.0,  # mmol/l # Total S2-
            "bc_s0": 0.0,  # mmol/l
            "bc_fe2": 0,  # wt% Fe2
            "bc_fe2_p": 0,  # wt% sorbed Fe2
            "bc_fe3": wt_percent_to_solid_conc(1, 56, 2.6, 0.65),  # wt% Fe
            "DB0": 4e-12 * 0,  # Bioturbation coefficient
            "DB_depth": 0,  # Bioturbation depth in m
            "BI0": 1e-6 * 0,  # should be < 1e-5
            "BI_depth": 0.0,  # Irrigation depth (0 = off)
            "eps": 1e-8,  # limiters
            "relax": 0.1,  # use 0.1 for coupled solver, and 0.8 otherwise
            "tolerance": 1e-5,  # convergence criterion
            "dt_tolerance": 0.1,  # steady state threshold
            "dt_target_change": 1.0,  # desired change per time step
            "solver_backend": "default",
            "dt_max": Q_("100 years").to("seconds").magnitude,  # mad dt step in years
            "dt_min": Q_("1 second").to("seconds").magnitude,  # min dt in years
            "dt_init": Q_("1 year").to("seconds").magnitude,  # initial dt
            "t_end": Q_("10 kyear").to("seconds").magnitude,  # max model time
            "max_steps": 2000,  # max number of iterations
            "VCDT": 0.044162589,  # VCDT reference ratio
            "initial_spacing": 0.01,  # meters
            "reaction_zone_spacing": 0.001,  # meters
            "reaction_zone": (5e-2, 2e-1),  # in meters
            "max_spacing": 0.1,  # meters, None = no cap
            "state_data": "state_data.npz",
            "title": None,  # defaults to current time
        }
    )

    mp.update(p_dict)
    _k1, k = get_reaction_constants(mp.phi, mp.pH)
    k = data_container(k)
    mp.bc_so4_32 = get_l_mass(mp.bc_so4, mp.so4_d, mp.VCDT)
    mp.bc_ts2_32 = get_l_mass(mp.bc_ts2, 0.0, mp.VCDT)  # Assume 0 delta for bc_h2s

    # -----------------------------------------------------------------------------
    # 2. MESH GENERATION (Variable Grid)
    # -----------------------------------------------------------------------------
    # mesh, z = make_grid(mp.max_depth, mp.initial_spacing, mp.max_spacing)
    mesh, z = make_grid2(
        mp.max_depth,
        mp.initial_spacing,
        mp.reaction_zone_spacing,
        mp.max_spacing,
        mp.reaction_zone,
    )
    mp.grid_points = len(z)

    # -----------------------------------------------------------------------------
    # 3. VARIABLES & DIFFUSION PROFILES
    # -----------------------------------------------------------------------------
    # Species that are part of the transport system
    species_list_partial = [
        "so4",
        "so4_32",
        "ts2",  # Total S2-
        "ts2_32",  # Total S2- 32S
        "o2",
        "poc",
        "fe2_total",
        "fe3",
        "fes",
        "fes_32",
        "s0",
        "s0_32",
        "fes2",
        "fes2_32",
    ]

    # Species that we use for reporting only
    report_species = [
        "fe2",
        "fe2_p",
        "hplus",
        "hs",  # HS-
        "hs_32",  # HS- 32S
        "h2s",
        "h2s_32",
    ]

    # these are not part of the T & R equation system
    species_list_full = species_list_partial + report_species

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

    # 2. Concentration Factors (Use these in Diagenetic Reactions)
    # [Fe2_pw] = [Fe2_total] * mp.f_pw_conc
    mp.fe2_pw_conc = 1.0 / R_factor

    # calculate H2S/HS- speciation
    pKa1 = 7.0
    Ka1 = 10 ** (-pKa1)
    H = 10 ** (-mp.pH)
    mp.h2s_frac = H / (H + Ka1)
    mp.hs_frac = Ka1 / (H + Ka1)

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
    D_mol.ts2 = diff_coeff(T_profile, 43.3, 0.85, mp.phi)
    D_mol.ts2_32 = D_mol.ts2
    D_mol.fe2 = diff_coeff(T_profile, 27.7, 1, mp.phi)
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
        "ts2": {"top": mp.bc_ts2, "type": "dissolved"},
        "ts2_32": {"top": mp.bc_ts2, "type": "dissolved"},
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

    # --- Setup LivePlotter ---
    plotter = None
    if plot_queue is None and mp.process_monitor != "none":
        output_path = f"{mp.plot_name}.pdf"
        import os

        video_path = (
            os.path.abspath(f"{mp.plot_name}.mp4")
            if mp.process_monitor == "video"
            else None
        )

        plotter = LivePlotter(
            layout_path=mp.layout_file,
            display_length=mp.display_length,
            output_path=output_path,
            video_path=video_path,
        )
        print(f"[Parent] Starting LivePlotter...", flush=True)
        plotter.start()
        print(f"[Parent] LivePlotter started. Queue: {plotter.queue}", flush=True)
        plot_queue = plotter.queue

    print(f"[Parent] Calling solver...", flush=True)
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
        plot_queue=plot_queue,
    )

    if plotter:
        plotter.stop()

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
