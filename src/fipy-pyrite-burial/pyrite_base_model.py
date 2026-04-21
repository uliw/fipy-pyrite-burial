"""
Define a reaction-transport model that computes pyrite precipitation.

as a function of organic matter availability, including isotopes.  Model units are
meter/second, concentrations are given mmol/liter (mol/m^3) and solids are expressed as
concentration per unit of solid volume (mmol/L_solid).

This keeps the physics of the "solid phase" independent of how much water is currently
squeezing around it.  If the sediment compacts (porosity ϕ decreases), the amount of
organic matter per gram of rock doesn't change, but the amount of organic matter per
liter of bulk sediment does.  ​

As such, a reaction between a liquid and a solid needs to be scaled

f = k * [SO4] * (1 - phi)/phi * [OM]

"""


def pyrite_model(p_dict: dict, plot_queue=None, experiment="pyrite"):
    """Model pyrite precipitation.

    As a function of organic matter availability, including isotopes
    Model units are meter/second, mmol/liter, and meter
    """
    # import numpy as np
    from fipy.tools import numerix as np
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

    # from reactions_new import diagenetic_reactions
    import reactions_new as rn
    from reaction_constants import get_reaction_constants
    from live_plot_lib import LivePlotter, capture_state
    import plot_data_new
    import pandas as pd

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    mp = data_container(
        {
            # -------- File Names & output --------------------
            "plot_name": f"{experiment}",
            "state_data": f"{experiment}_state.npz",
            "layout_file": "plot_layout.py",
            "process_monitor": "none",  # gui | video | none
            "process_monitor": "gui",  # gui | video | none
            "process_monitor": "video",  # gui | video | none
            "report_step": 2,  # how often to update plot
            "backup_step": 10000,  # create backups every nth step
            "title": None,  # defaults to current time
            "start_time": 0,  # i.e., when starting from a previous state
            # --------- Model Geometry & boundary conditions ------
            "max_depth": 2.0,  # meters
            "initial_spacing": 0.001,  # meters
            "reaction_zone_spacing": 0.001,  # meters
            "max_spacing": 0.1,  # meters, None = no cap
            "reaction_zone": (0.05, 0.8),  # in meters
            "isotopes": True,
            "temp": [10.0, 10.1],  # temp top, bottom, in C
            "w": Q_("0.2 cm/yr").to("m/s").m,  # sedimentation rate in m/s
            "advection": 0,  # upward directed flow component
            "pH": 7.5,  # porewater pH, Velde et al.
            "phi": 0.8,
            "so4_d": 21,  # seawater delta
            "msr_alpha": 1.07,  # MSR enrichment factor in mUr
            "hs_ox_alpha": 0.995,  # sulfide oxidation enrichment factor in mUr
            "s0_ox_alpha": 1,  # sulfide oxidation enrichment factor in mUr
            "bc_o2": 6,  # mmmol/l
            "bc_so4": 28.0,  # mmol/l
            "bc_ts2": 0.0,  # mmol/l # Total S2-
            "bc_s0": 0.0,  # mmol/l
            "bc_om": Q_("548 umol/(cm^2 * year)").to("mol/(m^2 * second)").magnitude,
            "bc_fe2": 0,  # wt% Fe2
            "bc_fe2_p": 0,  # wt% sorbed Fe2
            "bc_fe3": Q_("12 umol/(cm^2 * year)").to("mol/(m^2 * second)").magnitude,
            "DB0": Q_("0.2 cm^2/year").to("m^2/second").magnitude * 0,
            "DB_depth": 0,  # Bioturbation depth in m
            "BI0": 1e-6 * 0,  # should be < 1e-5
            "BI_depth": 0.0,  # Irrigation depth (0 = off)
            "dispro_so4_alpha": 1.02,  # about +20 mUr
            "dispro_hs_alpha": 9.0993,  # about -7 mUr
            "dispro_so4_hs_split": 0.5,  # i.e. 2 parts SO4, 1 part H2S
            # --------- Solver Parameters ----------------------------
            "max_steps": 20,  # max number of iterations
            "t_end": Q_("1 kyr").to("seconds").magnitude,
            "dt_min": Q_("1 minute").to("seconds").magnitude,  # time step in years
            "dt_init": Q_("1 month").to("seconds").magnitude,  # initial dt
            "dt_max": Q_("1 year").to("seconds").magnitude,  # time step in years
            "tolerance": 1e-12,  # convergence criterion
            "dt_tolerance": 1e-12,  # steady state threshold (stop simulation)
            "dt_target_change": 100,  # target change per step (for dt adaptation)
            "solver_backend": "default",  # see solver_calls for options
            "solver_backend": "LinearGMRESSolver",  # see solver_calls for options
            # ---------  Other --------------------------------------------------
            "eps": 1e-8,  # limiters
            "current_dt": 0.0,  # place holder
            "VCDT": 0.044162589,  # VCDT reference ratio
            "display_length": 2,  #
        }
    )

    # add reactions as needed
    p_dict["diagenetic_reactions"] = [
        rn.aerobic_respiration,
        rn.sulfate_reduction,
        rn.hs_oxidation,
        rn.elemental_sulfur_oxidation,
        rn.sulfide_mediated_iron_reduction,
        rn.fe2_oxidation,
        rn.fes_precipitation_terminal,
        rn.fes_dissolution,
        rn.fes_oxidation,
        rn.pyrite_formation_s0,
        rn.pyrite_formation_fes_ts2,
        rn.pyrite_oxidation,
    ]

    p_dict["instantenous_reactions"] = [
        rn.fe2_sorption_clip,
        # rn.sulfide_speciation_clip,
    ]

    # update with values passed from calling program
    mp.update(p_dict)

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

    mp.phi = CellVariable(name="porosity", mesh=mesh, value=mp.phi)
    _k1, k = get_reaction_constants(mp.pH)
    k = data_container(k)
    mp.bc_so4_32 = get_l_mass(mp.bc_so4, mp.so4_d, mp.VCDT)
    mp.bc_ts2_32 = get_l_mass(mp.bc_ts2, 0.0, mp.VCDT)  # Assume 0 delta for bc_h2s

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
    mp.fac_s = mp.phi.value / (1.0 - mp.phi.value)

    # Fe2 sorption fraction. Since sorption is faster than transport we treat it as
    # instantenous, i.e. it is just a function of concentration K_ads = k.fe2_p_eq which
    # is unitless (Conc_solid_vol / Conc_liquid_vol)
    # Fe_tot is mmol / L_solid
    # k.fe2_p_eqv is (mmol/L_solid) / (mmol/L_pw)
    # The 'volume ratio' term
    vol_ratio = mp.phi.value / (1.0 - mp.phi.value)

    # fraction of Fe2+ in porewater mmol/L_pw
    mp.f_diss = 1 / (k.fe2_p_eq + vol_ratio)
    mp.fe2_pw_conc = mp.f_diss
    # fraction of Fe2+ in sediment as (mmol / L_solid)
    mp.f_sorb = k.fe2_p_eq * mp.f_diss

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
        / (1 - np.log(mp.phi.value**2))
    )

    # -- Bioturbation and Irrigation Profiles (Robust Sigmoid) --
    D_mol.D_irr = compute_bio_irrigation_alpha(z, mp.BI0, mp.BI_depth)
    D_mol.D_bio = compute_sigmoidal_db(z, mp.DB0, mp.DB_depth, 0.1)
    # lumped modeling of Fe2 liq and Fe2 adsorbed
    D_mol.fe2_total = D_mol.fe2 * mp.f_diss

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

        if props["type"] == "particulate":
            # For particulate species, top is a flux in mol/(m²·s) bulk.
            # The solver transport term is weighted by (1-φ), so the CellVariable
            # must hold a solid-phase concentration: C_solid = J / (w * (1-φ)).
            # Robin BC: J_in = (1-φ) * (v_burial * C_solid - D * dC_solid/dx)
            # Therefore: dC_solid/dx = (v_burial * C_solid - J_in/(1-φ)) / D
            phi_top = mp.phi.value[0]  # porosity at the top face
            J_solid = props["top"] / (
                1.0 - phi_top
            )  # convert bulk flux → solid-phase flux

            D_total = getattr(D_mol, species_name, 0.0) + D_mol.D_bio
            if not isinstance(D_total, CellVariable):
                D_total = CellVariable(mesh=mesh, value=D_total)

            d_left = D_total.faceValue[mesh.facesLeft.value][0]
            if d_left > 1e-20:
                var.faceGrad.constrain(
                    [(mp.w * var.faceValue - J_solid) / D_total.faceValue],
                    mesh.facesLeft,
                )
            else:
                # Pure advection -> Dirichlet C_solid = J_solid / w
                val = J_solid / mp.w if mp.w > 0 else 0.0
                var.setValue(val)
                var.constrain(val, mesh.facesLeft)
        else:
            var.setValue(props["top"])
            var.constrain(props["top"], mesh.facesLeft)

        var.faceGrad.constrain([0.0], mesh.facesRight)

    if mp.state_data:
        print(f"Reading state from {mp.state_data}")
        read_state(c, mp.state_data)

    check_peclet_numbers(mesh, mp, D_mol, species_list_partial, bc_map)

    plotter = None
    if plot_queue is None and mp.process_monitor != "none":
        output_path = f"{mp.plot_name}.pdf"
        import os

        # For "gui", we also want video output
        video_path = None
        if mp.process_monitor in ["video", "gui"]:
            video_path = os.path.abspath(f"{mp.plot_name}.mp4")

        gui_enabled = mp.process_monitor == "gui"

        plotter = LivePlotter(
            layout_path=mp.layout_file,
            display_length=mp.display_length,
            output_path=output_path,
            video_path=video_path,
            gui=gui_enabled,
        )
        print(f"[Parent] Starting LivePlotter (gui={gui_enabled})...", flush=True)
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
        rn.diagenetic_reactions,
        mesh,
        D_mol,
        bc_map,
        z,
        plot_queue=plot_queue,
    )

    if plotter:
        plotter.stop()
    elif mp.process_monitor == "none":
        # Produce final plot even if monitoring was disabled
        print(f"[Parent] Producing final PDF plot: {mp.plot_name}.pdf")
        final_data = capture_state(
            mp,
            c,
            k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            current_dt=0.0,  # dt doesn't matter for final static plot
        )
        final_df = pd.DataFrame(final_data)
        plt_desc = plot_data_new.load_layout_from_file(final_df, mp.layout_file)
        plot_data_new.plot(
            final_df,
            mp.display_length,
            outfile=f"{mp.plot_name}.pdf",
            show=False,
            plot_description=plt_desc,
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
        rn.diagenetic_reactions,
        converged,
        step,
        total_time,
    )
