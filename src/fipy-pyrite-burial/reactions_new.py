"""Define the reactions."""

import numpy as np
from diff_lib import calculate_k_iron_reduction


def diagenetic_reactions(mp, c, k, f):
    """
    Main orchestrator for diagenetic reactions.
    Calculates limiters, initializes matrices, and calls specific process functions.

    Porosity Handling (Divided Form):
    ---------------------------------
    This model solves the 'divided' form of the conservation equations, where the
    volume fractions (porosity phi or 1-phi) are divided out.

    Equation Form:
       dC/dt + v*grad(C) = D*grad^2(C) + R_divided

    The reaction rates R_base are typically defined per unit porewater (mol/L_pw/s).
    - For Liquid Species: R_divided = R_base
    - For Solid Species:  R_divided = R_base * (phi / (1 - phi))

    This scaling ensures that for a reaction consuming 1 mol of L and producing 1 mol of S,
    the total mass balance (mol/L_bulk) is preserved:
       d/dt(phi*C_L) + d/dt((1-phi)*C_S) = -phi*R_base + (1-phi)*(R_base * phi / (1-phi)) = 0

    Consistency Note:
    The solver in diff_lib.py must NOT scale transport coefficients (v, D) by phi
    when using this divided source term logic (assuming constant phi).
    """
    from diff_lib import calculate_k_iron_reduction

    # 1. SETUP & INITIALIZATION
    # -------------------------
    species_list = list(c.keys())

    # Accumulators (The State)
    # LHS: Diagonal (Self) Coefficients (Implicit Sinks)
    LHS = {s: 0.0 for s in species_list}

    # CROSS: Off-Diagonal / Coupled Terms
    # Dict of List of Tuples: target -> [(source_var_name, coeff_value), ...]
    CROSS = {s: [] for s in species_list}

    RHS = {s: np.zeros_like(c.so4) for s in species_list}
    RATES = {s: np.zeros_like(c.so4) for s in species_list}

    # 2. CALCULATE LIMITERS
    # ---------------------
    eps = mp.eps
    limiters = {}

    # O2 Inhibition (1.0 -> 0.0)
    limiters["inhib_o2"] = eps / (c.o2 + eps)

    # Sulfate Limiter (Implicit 1/[S+K] and Explicit [S]/[S+K])
    K_so4 = 0.2
    limiters["so4_implicit"] = 1.0 / (c.so4 + K_so4)
    limiters["so4_32_implicit"] = 1.0 / (c.so4_32 + K_so4)

    limiters["so4_explicit"] = c.so4 / (c.so4 + K_so4)
    limiters["so4_32_explicit"] = c.so4_32 / (c.so4_32 + K_so4)

    limiters["fe3_explicit"] = 1.0  # c.fe3 / (c.fe3 + 1e-3)
    limiters["fe3_implicit"] = 1.0  # 1.0 / (c.fe3 + 1e-3)

    limiters["fes_explicit"] = c.fes / (c.fes + 1e-6)
    limiters["fes_implicit"] = 1 / (c.fes + 1e-6)

    K_alpha = 0.2
    limiters["alpha_explicit"] = c.so4 / (c.so4 + K_alpha)
    limiters["alpha_implicit"] = 1.0 / (c.so4 + K_alpha)

    # H2S Alpha Limiter (prevents numerical issues at trace concentrations)
    limiters["h2s_alpha_explicit"] = c.h2s.value / (c.h2s.value + 0.05)

    # update k-values
    k.fe3_h2s = calculate_k_iron_reduction(c.fe3, c.h2s)

    # 3. RUN PROCESSES
    # ----------------
    # Each function updates LHS, RHS, and RATES in place

    aerobic_respiration(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    sulfate_reduction(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    h2s_oxidation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    iron_reduction_h2s_lumped(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    fe2_adsoption_lumped(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    fe2_oxidation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    fes_dissolution(c, k, limiters, LHS, RHS, RATES, CROSS, mp)  #
    # fes_formation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)  #
    # equilibrate_fes_precipitation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)  #
    # fes_oxidation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    # pyrite_oxidation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    # pyrite_formation_s0(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    # pyrite_formation_fes_h2s(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    # ---- old ----
    # fe2_sorption(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    # iron_reduction_h2s(c, k, limiters, LHS, RHS, RATES, CROSS, mp)

    # 4. FINALIZE
    # -----------
    # Pack results into f container
    for s in species_list:
        setattr(f, s, (LHS[s], RHS[s], RATES[s], CROSS[s]))

    return f, RATES


# =============================================================================
# HELPER FUNCTIONS (Matrix Math Abstraction)
# =============================================================================


def add_implicit_sink(LHS, RATES, species, coeff, rate):
    """Add a consumption term to the LHS matrix.

    Add a consumption term to the LHS matrix.
    LHS = -Coefficient * Identity
    """
    # Use standard assignment to avoid in-place operator issues with some libraries
    LHS[species] = LHS[species] - coeff
    RATES[species] -= getattr(rate, "value", rate)


def add_explicit_source(RHS, RATES, species, rate):
    """Add a production term to the RHS vector.

    Add a production term to the RHS vector.
    RHS = -Rate (Standard library quirk for production)
    """
    RHS[species] = RHS[species] + rate
    RATES[species] += getattr(rate, "value", rate)


def add_implicit_coupling(CROSS, RATES, target_species, source_species, coeff, rate):
    """
    Add a coupled source term.

    If d[Target]/dt = +coeff * [Source]
    Then we add `ImplicitSourceTerm(coeff=coeff, var=Source)` to Target's equation.

    CROSS[target].append( (source, coeff) )

    as well as the associated implicit sink.
    """
    # FIXME: This needs the correct conversion factors for porosity mp.fac

    CROSS[target_species].append((source_species, coeff))
    # Note: Rates are accumulating scalar values for reporting, usually calculated explicitly before calling
    RATES[target_species] += getattr(rate, "value", rate)


def add_implicit_coupling_new(
    ctype, CROSS, RATES, LHS, target_species, source_species, coeff, rate, mp
):
    """
    Add a coupled source term with porosity correction.

    If d[Target]/dt = +coeff * [Source]
    Then we add `ImplicitSourceTerm(coeff=coeff*fac, var=Source)` to Target's equation.

    ctype: connection type ('l2l', 'l2s', 's2s', 's2l')
    CROSS: Off-diagonal coupling matrix
    RATES: Rate reporting dictionary
    LHS: Diagonal matrix (implicit sinks)
    mp: Model parameters (contains fac_s)
    """
    # 1. Determine Porosity Correction Factor
    # mp.fac_s = phi / (1-phi)
    if ctype == "liquid_2_liquid":  # liquid to liquid
        fac = 1.0
    elif ctype == "liquid_2_solid":  # liquid to solid
        fac = mp.fac_s
    elif ctype == "solid_2_solid":  # solid to solid
        fac = 1.0
    elif ctype == "solid_2_liquid":  # solid to liquid
        fac = 1.0 / mp.fac_s
    else:
        raise ValueError(f"type must be l2l, l2s, s2s, s2l, not {ctype}")

    # 2. Add Coupling (Implicit Source in Target's Equation)
    # The coefficient in the Target's equation is coeff * fac
    CROSS[target_species].append((source_species, coeff * fac))

    # 3. Add Implicit Sink (in Source's Equation)
    # The reaction consumes 'source_species' at rate 'coeff * source_species'
    # Note: add_implicit_sink handles negative sign: LHS = LHS - coeff
    add_implicit_sink(LHS, RATES, source_species, coeff, rate)

    # 4. Update Rate Reporting for Target
    # We report the rate as seen by the target species (including fac)
    RATES[target_species] += getattr(rate, "value", rate) * fac


# =============================================================================
# PROCESS FUNCTIONS (The Biogeochemistry)
# =============================================================================
def aerobic_respiration(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define POC consumption by aerobic respiration."""
    rate_base = k.poc_o2 * c.poc * c.o2

    # POC Sink - SOLID
    coeff_poc = k.poc_o2 * c.o2 * mp.fac_s
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, rate_base * mp.fac_s)

    # O2 Sink (1.27x) - LIQUID
    coeff_o2 = 1.27 * k.poc_o2 * c.poc
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, 1.27 * rate_base)
    # No produced species here (CO2 ignored)


def sulfate_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate sulfate reduction.

    Reaction: 2 POC + 1 SO4 -> 1 H2S Ref: POC (k.poc_so4)
    """
    # 1. Base Rate
    poc_rate = k.poc_so4 * c.poc * c.so4 * lim["so4_implicit"] * lim["inhib_o2"]
    so4_rate = poc_rate * 0.5

    # 2. POC Sink (Ref Species) - SOLID
    coeff_poc = k.poc_so4 * lim["so4_explicit"] * lim["inhib_o2"] * mp.fac_s
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, poc_rate * mp.fac_s)

    # 3. SO4 Sink -> Rate = 0.5 * Base - LIQUID
    coeff_so4 = k.poc_so4 * c.poc * lim["inhib_o2"] * lim["so4_implicit"] * 0.5

    # 4. Sulfate reduction
    add_implicit_coupling_new(
        "liquid_2_liquid",  # type
        CROSS,  #  Off-diagonal coupling matrix
        RATES,  #  Rate reporting dictionary
        LHS,  # Diagonal matrix (implicit sinks)
        "h2s",  # species that is produced
        "so4",  # species that is consumed
        coeff_so4,  # reaction coefficient
        so4_rate,  # coeff * concentration
        mp,  # model parameters
    )

    # isotopes
    if hasattr(c, "so4_32"):
        alpha = 1.0 + (mp.msr_alpha - 1.0) * lim["alpha_explicit"]
        s_val = c.so4 + 1e-12
        s32_val = c.so4_32 + 1e-12
        f_32 = alpha / (s_val + (alpha - 1) * s32_val + 1e-30)
        coeff_so4_32 = f_32 * so4_rate

        add_implicit_coupling_new(
            "liquid_2_liquid",  # type
            CROSS,
            RATES,
            LHS,
            "h2s_32",  # species that is produced
            "so4_32",  # source species
            coeff_so4_32,  # implicit coeff for sink
            coeff_so4_32 * c.so4_32,  # explicit rate for reporting
            mp,
        )

        # # sulfate 32
        # add_implicit_sink(LHS, RATES, "so4_32", coeff_so4_32, coeff_so4_32 * c.so4_32)
        # add_implicit_coupling(
        #     CROSS,
        #     RATES,
        #     "h2s_32",  # species that is produced
        #     "so4_32",  # source species
        #     coeff_so4_32,  # implicit coeff for sink
        #     coeff_so4_32 * c.so4_32,  # explicit rate for reporting
        # )


def h2s_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 H2S + 0.5 O2 -> 1 S0"""
    # H2S Sink - LIQUID
    # Ref: H2S
    coeff_h2s = k.h2s_ox * c.o2

    # O2 Sink (0.5x) - LIQUID
    coeff_o2 = 0.5 * k.h2s_ox * c.h2s
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2)

    # S0 Source (1.0x) - SOLID, Couple to H2S
    add_implicit_coupling_new(
        "liquid_2_solid",
        CROSS,
        RATES,
        LHS,
        "s0",  # species that is produced
        "h2s",  # source species
        coeff_h2s,  # implicit coeff for sink
        coeff_h2s * c.h2s,  # explicit rate for reporting
        mp,
    )

    if hasattr(c, "h2s_32"):
        """Calculate Fractionation Factors using Explicit Values (Linearization)
        Note: We use .value to get numpy arrays for the denominator to avoid
        creating complex non-linear FiPy terms that slow convergence.
         FIX 1: Use coeff_h2s (Total Coeff) as the base
        FIX 2: Do NOT divide by c.h2s_32, this is because
        f32 = f * a  * s32 / (s + (a-1) * s32)
        and f32 = coeff32 * s
        now we substitute these terms for f32 on both sides:
        coeff_32 * s32 = coeff_s * s * a * s32/ (s + (a-1) * s32)
        -> s32 appears on both sides of teh equation, so they cancel!
        solution: remove s32 on the right hand side
        f32 =  coeff_s * s * a * s32/ (s + (a-1))
        """
        alpha = 1.0 + (mp.h2s_ox_alpha - 1.0) * lim["h2s_alpha_explicit"]

        s_val = c.h2s + 1e-20
        s32_val = c.h2s_32 + 1e-20
        denom = s_val + (alpha - 1.0) * s32_val

        # Scaling factor for the coefficient
        # Logic: Coeff_32 = Coeff_Tot * (S_Tot * alpha / Denom)
        # We use c.h2s (Variable) for S_Tot to keep the Jacobian accurate
        scaling_factor = (c.h2s * alpha) / denom
        coeff_h2s_32 = coeff_h2s * scaling_factor

        # S0_32 coupled to H2S_32
        add_implicit_coupling_new(
            "liquid_2_solid",
            CROSS,
            RATES,
            LHS,
            "s0_32",  # species that is produced
            "h2s_32",  # source species
            coeff_h2s_32,  # implicit coeff for sink
            coeff_h2s_32 * c.h2s_32,  # explicit rate for reporting
            mp,
        )


def iron_reduction_h2s_lumped(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define iron reduction by h2s.
    1 Fe3 + 0.5 H2S -> 1 Fe2 + 0.5 S0
    Fe2+ is a liquid!
    """
    k.fe3_h2s = calculate_k_iron_reduction(c.fe3, c.h2s)

    # 1. Fe3 Sink - SOLID (Linear)
    # Rate = k * [H2S] * [Fe3]
    coeff_fe3 = k.fe3_h2s * c.h2s
    add_implicit_sink(LHS, RATES, "fe3", coeff_fe3, coeff_fe3 * c.fe3)

    # 2. H2S Sink (0.5x) - LIQUID (Linear)
    coeff_h2s = k.fe3_h2s * c.fe3 * 0.5

    # 3. Fe2+ Source (1.0x) - Liquid (Coupled to Fe3)
    # MUST BE LINEAR to match Fe3 Sink
    # Rate = k * [H2S] * [Fe3]
    # Coeff = k * [H2S]

    coeff_coupling_fe2 = k.fe3_h2s * c.h2s

    add_implicit_coupling_new(
        "solid_2_liquid",
        CROSS,
        RATES,
        LHS,
        "fe2_total",  # target
        "fe3",  # source
        coeff_coupling_fe2,  # implicit coefficient
        coeff_coupling_fe2 * c.fe3,  # explicit rate
        mp,
    )

    # 4. Elemental sulfur - Solid (Coupled to H2S)
    # Rate = 0.5 * k * [Fe3] * [H2S] * mp.fac_s
    # Matches H2S Sink stoichiometry and kinetics exactly
    add_implicit_coupling_new(
        "liquid_2_solid",
        CROSS,
        RATES,
        LHS,
        "s0",
        "h2s",
        coeff_h2s,
        coeff_h2s * c.h2s,
        mp,
    )

    if hasattr(c, "h2s_32"):
        # Elemental sulfur 32S - Solid
        add_implicit_coupling_new(
            "liquid_2_solid",
            CROSS,
            RATES,
            LHS,
            "s0_32",
            "h2s_32",
            coeff_h2s,
            coeff_h2s * c.h2s_32,
            mp,
        )


def fe2_adsoption_lumped(c, k, lim, LHS, RHS, RATES, Cross, mp):
    """
    Handle Iron Partitioning algebraically.
    Instead of calculating rates, we calculate fractions.

    System State: 'fe_total' is the primary variable.
    fe2 (liquid) and fe2_p (solid) are derived helper views.
    """

    # -----------------------------------------------------------
    # RECONSTRUCT SPECIES FOR OTHER REACTIONS
    # -----------------------------------------------------------
    # Other reactions (Pyrite precip, etc.) need [Fe2] and [Fe2_p].
    # We essentially 'distribute' the total iron to these views.
    # NOTE: c.fe2 and c.fe2_p must be updated so subsequent functions
    # (like iron_sulfide_formation) read the correct values.

    # This might require c to be mutable or update the variables in place.
    # In FiPy, we can't easily overwrite 'c.fe2' if it's the solution variable.
    # STRATEGY:
    # 1. Solve for 'fe_total' in the main solver.
    # 2. Inside this function, calculate fe2 and fe2_p from fe_total.
    # 3. Store them in 'c' so subsequent reactions use them.

    if hasattr(c, "fe2_total"):
        c.fe2.setValue(c.fe2_total * mp.f_diss)
        c.fe2_p.setValue(c.fe2_total * mp.f_sorb)


def fe2_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Reaction: 4 Fe2+ O2 -> 1 Fe3OOH
    Note: Fe2_total tracks Fe2 liquid and sorbed. However the
    reaction rates are the same, so we use fe2_total
    """
    rate_base = k.fe2_ox * c.fe2_total * c.o2

    # Fe2+ Sink - Liquid
    coeff_fe2 = k.fe2_ox * c.o2

    # O2 Sink (1/4) - LIQUID
    coeff_o2 = k.fe2_ox * c.fe2_total * 0.25
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, rate_base * 0.25)

    # Fe3 Source (1.0x) - SOLID
    # Couple to Fe2 (Liquid)
    # Fe3 is solid (mp.fac_s scaling needed).
    # d((1-phi)Fe3)/dt = phi * k * Fe2 * O2.
    # coeff * (1-phi) = phi * k * O2.
    # coeff = mp.fac_s * k * O2.
    add_implicit_coupling_new(
        "solid_2_solid",
        CROSS,
        RATES,
        LHS,
        "fe3",  # product
        "fe2_total",  # source
        k.fe2_ox * c.o2,  # coefficient
        rate_base,  # rate for reporting
        mp,
    )


def fes_dissolution(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """FeS dissolution only.

    Precipitation is being handled by equilibrate_fes_precipitation
    """
    # 1. Current State
    fe2_val = c.fe2_total.value
    fe2_liq_val = fe2_val * mp.f_diss  # Bulk moles in liquid
    h2s_val = c.h2s.value
    fes_val = c.fes.value  # mol/L_solid

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_liq_val * h2s_val) / omega_den

    # Derivatives (Slopes)
    deriv_fe2 = (k.fes_isp * mp.f_diss * h2s_val) / omega_den
    deriv_h2s = (k.fes_isp * mp.f_diss * fe2_val) / omega_den

    # ----- Dissolution Logic (Omega < 1) ------- #
    is_diss = (omega <= 1.0).astype(float)
    epsilon_fes = 1e-10
    fes_limiter = fes_val / (fes_val + epsilon_fes)
    # coeff_diss in 1/s (frequency of solid dissolution)
    coeff_diss = k.fes_isd * (1.0 - omega) * is_diss * fes_limiter

    # Sink for FeS (Solid)
    add_implicit_sink(LHS, RATES, "fes", coeff_diss, coeff_diss * fes_val)

    # Source for Fe2_total (Bulk)
    # Rate_bulk = Rate_solid * (1 - phi)
    add_implicit_coupling(
        CROSS,
        RATES,
        "fe2_total",
        "fes",
        coeff_diss * (1.0 - mp.phi),
        coeff_diss * fes_val * (1.0 - mp.phi),
    )

    # Source for H2S (Porewater)
    # Rate_pw = Rate_solid * (1 - phi) / phi = Rate_solid / fac_s
    add_implicit_coupling(
        CROSS,
        RATES,
        "h2s",
        "fes",
        coeff_diss / mp.fac_s,
        coeff_diss * fes_val / mp.fac_s,
    )

    # --- 7. Isotopes (32S) ---
    if hasattr(c, "h2s_32"):
        h2s_32_val = c.h2s_32.value
        f32_h2s = h2s_32_val / (h2s_val + 1e-20)

        # Dissolution 32S (Solid Sink)
        add_implicit_sink(LHS, RATES, "fes_32", coeff_diss, coeff_diss * c.fes_32.value)

        # Dissolution 32S (Liquid Source)
        add_implicit_coupling(
            CROSS,
            RATES,
            "h2s_32",
            "fes_32",
            coeff_diss / mp.fac_s,
            coeff_diss * c.fes_32.value / mp.fac_s,
        )


def fes_formation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """FeS precipitation and dissolution.

    This requires timesteps of 1 minute or less.
    """
    # 1. Current State
    fe2_val = c.fe2_total.value
    fe2_liq_val = fe2_val * mp.f_diss  # Bulk moles in liquid
    h2s_val = c.h2s.value
    fes_val = c.fes.value  # mol/L_solid

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_liq_val * h2s_val) / omega_den

    # 3. Precipitation Logic (Omega > 1)
    is_precip = (omega > 1.0).astype(float)
    rate_precip_total = k.fes_isp * (omega - 1.0) * is_precip  # mol/L_bulk/s

    # Derivatives (Slopes)
    deriv_fe2 = (k.fes_isp * mp.f_diss * h2s_val) / omega_den
    deriv_h2s = (k.fes_isp * mp.f_diss * fe2_val) / omega_den

    # --- Fe2_total Equation (Bulk) ---
    l_fe2 = deriv_fe2
    r_fe2 = rate_precip_total - (deriv_fe2 * fe2_val)
    add_implicit_sink(LHS, RATES, "fe2_total", l_fe2, rate_precip_total)
    add_explicit_source(RHS, RATES, "fe2_total", -r_fe2)

    # --- H2S Equation (Porewater) ---
    # Conversion: Rate_pw = Rate_bulk / phi
    l_h2s = deriv_h2s / mp.phi
    r_h2s = (rate_precip_total / mp.phi) - (l_h2s * h2s_val)
    add_implicit_sink(LHS, RATES, "h2s", l_h2s, rate_precip_total / mp.phi)
    add_explicit_source(RHS, RATES, "h2s", -r_h2s)

    # --- FeS (Solid) Accumulation ---
    # Conversion: Rate_solid = Rate_bulk / (1 - phi)
    # Since deriv_fe2 is dR_bulk/dFe2_bulk:
    l_fes_precip = deriv_fe2 / (1.0 - mp.phi)
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes",
        "fe2_total",
        l_fes_precip,
        (rate_precip_total / (1.0 - mp.phi)),
    )

    # ----- Dissolution Logic (Omega < 1) ------- #
    is_diss = (omega <= 1.0).astype(float)
    epsilon_fes = 1e-10
    fes_limiter = fes_val / (fes_val + epsilon_fes)
    # coeff_diss in 1/s (frequency of solid dissolution)
    coeff_diss = k.fes_isd * (1.0 - omega) * is_diss * fes_limiter

    # Sink for FeS (Solid)
    add_implicit_sink(LHS, RATES, "fes", coeff_diss, coeff_diss * fes_val)

    # Source for Fe2_total (Bulk)
    # Rate_bulk = Rate_solid * (1 - phi)
    add_implicit_coupling(
        CROSS,
        RATES,
        "fe2_total",
        "fes",
        coeff_diss * (1.0 - mp.phi),
        coeff_diss * fes_val * (1.0 - mp.phi),
    )

    # Source for H2S (Porewater)
    # Rate_pw = Rate_solid * (1 - phi) / phi = Rate_solid / fac_s
    add_implicit_coupling(
        CROSS,
        RATES,
        "h2s",
        "fes",
        coeff_diss / mp.fac_s,
        coeff_diss * fes_val / mp.fac_s,
    )
    # At the end of fes_formation, fix rate reporting
    # _c_h2s np.float64(1.0862995735298113e-38) float64
    # _c_fes np.float64(7.060947227943774e-39) float64
    # _d np.float64(-0.0) float64
    # _l np.float64(0.0003639307504732167) float64
    # _p np.float64(0.0020109244098000007) float64
    # mp.phi 0.65 float
    net_fes_rate = (rate_precip_total - (coeff_diss * fes_val)) / (1.0 - mp.phi)
    RATES["fes"] = net_fes_rate  # Use setValue if RATES contains CellVariables
    i = 400
    _n = net_fes_rate[i]
    _p_total = rate_precip_total[i]  #  # mol/m^3_bulk/s
    _p_actual = l_fes_precip[i]
    _d = ((coeff_diss * fes_val) / (1.0 - mp.phi))[i]  # dissolution
    # correction factors
    _c_fes = -r_fe2[i]
    _c_h2s = -r_h2s[i]
    _fe2 = fe2_liq_val[i]
    _h2s = h2s_val[i]

    # --- 7. Isotopes (32S) ---
    if hasattr(c, "h2s_32"):
        h2s_32_val = c.h2s_32.value
        f32_h2s = h2s_32_val / (h2s_val + 1e-20)

        # Precipitation 32S
        # Use same porewater scaling as H2S total
        l_h2s_32 = l_h2s
        rate_32_precip = (rate_precip_total / mp.phi) * f32_h2s
        r_h2s_32 = rate_32_precip - (l_h2s_32 * h2s_32_val)

        add_implicit_sink(LHS, RATES, "h2s_32", l_h2s_32, rate_32_precip)
        add_explicit_source(RHS, RATES, "h2s_32", -r_h2s_32)

        # Accumulation in FeS_32 (Solid)
        # Scale: f32 * deriv_fe2 / (1-phi)
        l_fes_32_precip = (deriv_fe2 / (1.0 - mp.phi)) * f32_h2s
        add_implicit_coupling(
            CROSS,
            RATES,
            "fes_32",
            "fe2_total",
            l_fes_32_precip,
            (rate_32_precip * mp.fac_s),
        )

        # Dissolution 32S (Solid Sink)
        add_implicit_sink(LHS, RATES, "fes_32", coeff_diss, coeff_diss * c.fes_32.value)

        # Dissolution 32S (Liquid Source)
        add_implicit_coupling(
            CROSS,
            RATES,
            "h2s_32",
            "fes_32",
            coeff_diss / mp.fac_s,
            coeff_diss * c.fes_32.value / mp.fac_s,
        )


def fes_formation_old(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Model ironsulfide precipitation and dissolution.

    Reaction: Fe2 + H2S <-> FeS
    Here we use the fraction of Fe2 that is liquid.
    c.fe2_total is the lumped expression for Fe2 liquid + sorbed
    mp.f_diss is the Fe2 fraction that is dissolved

    The precipitation reaction is
    mp.fac_s * k.fes_isp * (c.fe2_liq * c.h2s/(c.hplus * k.fes_sp) -1)
    and dissolution is
    mp.fac_s * k.fes_isd * c.fes * (1 - c.fe2_liq * c.h2s/(c.hplus * k.fes_sp))

    Note that dissolution is much slower.
    """
    # 1. Get current values (numpy arrays for coefficients)
    fe2_val = c.fe2_total.value
    fe2_liq_val = fe2_val * mp.f_diss
    h2s_val = c.h2s.value

    # 2. Calculate Saturation State (Omega)
    # Omega = [Fe2+][H2S] / ([H+] * Ksp)
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_liq_val * h2s_val) / omega_den

    # 3. Switches
    is_precip = (omega > 1.0).astype(float)
    is_diss = (omega <= 1.0).astype(float)

    # 4. Precipitation Rates and Coefficients
    # --- PRECIPITATION (Omega > 1) ---

    # Raw Rate (Ideal): R = k_isp * (Omega - 1)
    # We check if this rate would consume more than 90% of fe2_liq in one 'unit' step

    # Note: fes_precip_rate_linear = k.fes_isp * (omega - 1.0) * is_precip
    # But for limiter calculation, we can just use the linearized magnitude
    # effectively: Rate ~ k_isp * Omega * is_precip (upper bound) or k_isp*(Omega-1)
    # Let's limit based on the actual net rate k_isp*(Omega-1) relative to Fe2_liq

    # Calculate raw net rate for checking
    raw_rate_net = k.fes_isp * (omega - 1.0) * is_precip
    raw_rate_net = np.maximum(raw_rate_net, 0.0)

    # Check for time step (Transient vs Steady State)
    # If mp.dt exists (Transient), we use it.
    # If not (Steady State), we default to 1.0 (assuming normalized rates or implicit handling)
    if hasattr(mp, "dt.max") and mp.dt_max is not None:
        dt_sim = mp.dt_max
    else:
        dt_sim = 1.0

    # Maximum allowed rate (90% of liquid iron OR H2S per time step)
    # R * dt < 0.9 * C  =>  R < 0.9 * C / dt
    # Check both reactants to avoid depleting either
    max_rate_fe2 = (0.9 * fe2_liq_val) / (dt_sim + 1e-30)
    max_rate_h2s = (0.9 * h2s_val) / (dt_sim + 1e-30)

    max_rate_allowed = np.minimum(max_rate_fe2, max_rate_h2s)

    # Calculate Limiting Factor alpha \in [0, 1]
    # Avoid division by zero
    limiter_alpha = np.minimum(1.0, max_rate_allowed / (raw_rate_net + 1e-30))

    # Apply limiter to the rate constant effect
    k_isp_eff = k.fes_isp * limiter_alpha

    # Common Factor for implicit terms: k_eff / Omega_den
    common_precip = k_isp_eff * is_precip / omega_den

    # 5. Dissolution Rates and Coefficients
    # --- DISSOLUTION (Omega <= 1) ---
    # Rate_base = k.fes_isd * (1 - Omega) * FeS
    # We treat Omega as explicit (linearize against FeS only)
    omega_diss = np.minimum(omega, 1.0)
    fes_diss_coeff = k.fes_isd * (1.0 - omega_diss) * is_diss * lim["fes_explicit"]

    # 6. Apply terms to Solver matrices

    # --- Fe2_total (Liquid) ---
    # Sink (Precipitation): -k * Omega = -(k * H2S / Omega_den) * fe2_liq
    # Since fe2_liq = fe2_total * f_diss:
    # coeff = (k * H2S / Omega_den) * f_diss
    coeff_fe2_sink = common_precip * h2s_val * mp.f_diss
    add_implicit_sink(LHS, RATES, "fe2_total", coeff_fe2_sink, coeff_fe2_sink * fe2_val)

    # Source (Dissolution): +k_isd * (1-Omega) * FeS [Liquid Source = Solid Rate]
    add_implicit_coupling(
        CROSS, RATES, "fe2_total", "fes", fes_diss_coeff, fes_diss_coeff * c.fes.value
    )

    # --- H2S (Liquid) ---
    # Sink (Precipitation): -k * Omega = -(k * fe2_liq / Omega_den) * H2S
    coeff_h2s_sink = common_precip * fe2_liq_val
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s_sink, coeff_h2s_sink * h2s_val)

    # Source (): +k
    add_explicit_source(RHS, RATES, "h2s", k_isp_eff * is_precip)

    # Source (Dissolution): +k_isd * (1-Omega) * FeS
    add_implicit_coupling(
        CROSS, RATES, "h2s", "fes", fes_diss_coeff, fes_diss_coeff * c.fes.value
    )

    # --- FeS (Solid) ---
    # Source (Precipitation): +k * Omega * fac_s = +(k * H2S * f_diss / Omega_den) * fe2_total * fac_s
    coeff_fes_source = common_precip * h2s_val * mp.f_diss * mp.fac_s
    add_implicit_coupling(
        CROSS, RATES, "fes", "fe2_total", coeff_fes_source, coeff_fes_source * fe2_val
    )

    # Sink (Precipitation Correction): -k * fac_s
    add_explicit_source(RHS, RATES, "fes", -k_isp_eff * is_precip * mp.fac_s)

    # Sink (Dissolution): -k_isd * (1-Omega) * FeS * fac_s
    coeff_fes_sink = fes_diss_coeff * mp.fac_s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes_sink, coeff_fes_sink * c.fes.value)

    # 7. Isotopes (32S)
    if hasattr(c, "h2s_32"):
        # We assume no fractionation for FeS formation/dissolution (alpha = 1.0)
        # Precipitation Rate_32 = Rate_Base * (H2S_32 / H2S)
        h2s_inv = 1.0 / (h2s_val + 1e-20)

        # Sink for H2S_32 (Precipitation):
        # Rate = [k * Omega / H2S - k / H2S] * H2S_32
        # coeff = k * (fe2_liq / Omega_den - 1 / H2S)
        coeff_h2s_32_precip = (
            k_isp_eff * (fe2_liq_val / omega_den - h2s_inv) * is_precip
        )
        coeff_h2s_32_precip = np.maximum(coeff_h2s_32_precip, 0.0)
        add_implicit_sink(
            LHS,
            RATES,
            "h2s_32",
            coeff_h2s_32_precip,
            coeff_h2s_32_precip * c.h2s_32.value,
        )

        # Source for FeS_32 (Precipitation, coupled to H2S_32):
        add_implicit_coupling(
            CROSS,
            RATES,
            "fes_32",
            "h2s_32",
            coeff_h2s_32_precip * mp.fac_s,
            coeff_h2s_32_precip * c.h2s_32.value * mp.fac_s,
        )

        # --- DISSOLUTION ---
        # Sink for FeS_32:
        # Rate = k_isd * (1-Omega) * FeS_32 * fac_s
        add_implicit_sink(
            LHS, RATES, "fes_32", coeff_fes_sink, coeff_fes_sink * c.fes_32.value
        )

        # Source for H2S_32 (coupled to FeS_32):
        # Rate = k_isd * (1-Omega) * FeS_32
        add_implicit_coupling(
            CROSS,
            RATES,
            "h2s_32",
            "fes_32",
            fes_diss_coeff,
            fes_diss_coeff * c.fes_32.value,
        )


def fes_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 2.25 O2 -> 1 Fe3 + 1 SO4"""
    rate_base = k.fes_ox * c.fes * c.o2
    rate_base_32 = k.fes_ox * c.fes_32 * c.o2

    # FeS Sink - SOLID
    coeff_fes = k.fes_ox * c.o2 * mp.fac_s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, rate_base * mp.fac_s)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, rate_base_32 * mp.fac_s)

    # O2 Sink (2.25x) - LIQUID
    # Depends on FeS (Solid).
    coeff_o2_fes = 2.25 * k.fes_ox * c.fes
    # Wait, explicit sink for O2? O2 is liquid.
    # Rate = 2.25 * k * O2 * FeS.
    # Implicit Sink for O2: coeff = 2.25 * k * FeS.
    add_implicit_sink(LHS, RATES, "o2", coeff_o2_fes, rate_base * 2.25)

    # Fe3 Source (1.0x) - SOLID
    # Couple to FeS (Solid).
    # Rate = k * FeS * O2.
    # d((1-phi)Fe3)/dt = phi * (k * FeS * O2) -> NO!
    # FeS is solid. Rate is usually k_solid * FeS * O2.
    # If k is porewater based:
    # Scale appropriately.
    # If Fe3 and FeS are both solid, they share the same volume scalings.
    # coeff = k * O2. (Assuming implicit coupling will add coeff * FeS).
    # Let's check diff_lib.
    # Target Solid: coeff * (1-phi) * FeS.
    # We want d((1-phi)Fe3) = ... + (1-phi) * k * FeS * O2 ??
    # Usually: dC/dt (solid) = Rate (solid).
    # If Rate is k * FeS * O2.
    # Coeff = k * O2.
    # If we use mp.fac_sLogic?
    # Let's assume consistent k units.
    # Reactions involving solids are usually scaled by (1-phi) in the solver logic for "divided" equations?
    # No, 'divided' means we modeled C, not (1-phi)C.
    # But `diff_lib` multiplies by `scaling`.
    # So `dC/dt = ... + k*C`.
    # So `coeff = k * O2`.
    # Wait, in the sink, we used `coeff_fes * mp.fac_s`.
    # Why? `diagenetic_reactions` doc says "For Solid Species: R_divided = R_base * (phi/(1-phi))".
    # This implies R_base is porewater rate.
    # If we couple FeS (Solid) -> Fe3 (Solid), and both use porewater rate base?
    # Then we need `mp.fac_s`.
    add_implicit_coupling(
        CROSS, RATES, "fe3", "fes", k.fes_ox * c.o2 * mp.fac_s, rate_base * mp.fac_s
    )

    # SO4 Source (1.0x) - LIQUID
    # Couple to FeS.
    # Target Liquid. No mp.fac_s.
    add_implicit_coupling(CROSS, RATES, "so4", "fes", k.fes_ox * c.o2, rate_base)
    add_implicit_coupling(
        CROSS, RATES, "so4_32", "fes_32", k.fes_ox * c.o2, rate_base_32
    )


def pyrite_formation_s0(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 1 S0 -> 1 FeS2"""
    # S0 Sink - SOLID
    coeff_s0 = k.fes_s0 * c.fes * mp.fac_s
    add_implicit_sink(LHS, RATES, "s0", coeff_s0, coeff_s0 * c.s0)
    add_implicit_sink(LHS, RATES, "s0_32", coeff_s0, coeff_s0 * c.s0_32)

    # FeS Sink (1.0x) - SOLID
    coeff_fes = k.fes_s0 * c.s0 * mp.fac_s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, coeff_fes * c.fes)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, coeff_fes * c.fes_32)

    # FeS2 Source (1.0x) - SOLID
    # Couple to FeS?
    # Rate = k * FeS * S0.
    # Couple to FeS: coeff = k * S0 * mp.fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2",
        "fes",
        k.fes_s0 * c.s0 * mp.fac_s,
        k.fes_s0 * c.s0 * c.fes * mp.fac_s,
    )

    # FeS2_32 Source
    # Sum of S0_32 and FeS_32 ?
    # FeS2 contains 2 sulfurs.
    # Rate FeS2_32 is purely tracking S32 mass.
    # S32 comes from S0_32 and FeS_32.
    # Couple to both!
    # Term 1: from S0_32. Rate = k * FeS * S0_32. Coeff = k * FeS * mp.fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "s0_32",
        k.fes_s0 * c.fes * mp.fac_s,
        k.fes_s0 * c.fes * c.s0_32 * mp.fac_s,
    )
    # Term 2: from FeS_32. Rate = k * S0 * FeS_32. Coeff = k * S0 * mp.fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "fes_32",
        k.fes_s0 * c.s0 * mp.fac_s,
        k.fes_s0 * c.s0 * c.fes_32 * mp.fac_s,
    )


def pyrite_formation_fes_h2s(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 1 H2S -> 1 FeS2"""
    # FeS Sink - SOLID
    coeff_fes = k.fes_h2s * c.h2s * mp.fac_s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, coeff_fes * c.fes)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, coeff_fes * c.fes_32)

    # H2S Sink (1.0x) - LIQUID
    coeff_h2s = k.fes_h2s * c.fes
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s_32)

    # FeS2 Source (1.0x) - SOLID
    # Couple to FeS.
    # Rate = k * H2S * FeS.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2",
        "fes",
        k.fes_h2s * c.h2s * mp.fac_s,
        k.fes_h2s * c.h2s * c.fes * mp.fac_s,
    )

    # FeS2_32 Source
    # From FeS_32 and H2S_32.
    # Couple to FeS_32
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "fes_32",
        k.fes_h2s * c.h2s * mp.fac_s,
        k.fes_h2s * c.h2s * c.fes_32 * mp.fac_s,
    )
    # Couple to H2S_32
    # Rate = k * FeS * H2S_32.
    # Target Solid, Source Liquid. Coeff Needs mp.fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "h2s_32",
        mp.fac_s * k.fes_h2s * c.fes,
        mp.fac_s * k.fes_h2s * c.fes * c.h2s_32,
    )


def apply_rate_limiter(rate, var, fraction=0.5, eps=1e-12):
    """Limit rate so it doesn't consume more than a fraction of available var."""
    val = var.value if hasattr(var, "value") else var
    max_rate = val * fraction / 1.0  # Normalized dt=1 for steady state sweep
    return np.minimum(rate, np.maximum(max_rate, 0.0))


def pyrite_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Reaction: 1 FeS2 + 3.5 O2 -> 1 Fe3 + 2 SO4
    Ref: FeS2 (k.fes2_ox)
    """
    # FeS2 Sink - SOLID
    coeff_fes2 = k.fes2_ox * c.o2 * mp.fac_s
    add_implicit_sink(LHS, RATES, "fes2", coeff_fes2, coeff_fes2 * c.fes2)
    add_implicit_sink(LHS, RATES, "fes2_32", coeff_fes2, coeff_fes2 * c.fes2_32)

    # O2 Sink (3.5x) - LIQUID
    coeff_o2 = 3.5 * k.fes2_ox * c.fes2
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2)

    # Fe3 Source (1.0x) - SOLID
    # Couple to FeS2
    add_implicit_coupling(
        CROSS,
        RATES,
        "fe3",
        "fes2",
        k.fes2_ox * c.o2 * mp.fac_s,
        k.fes2_ox * c.o2 * c.fes2 * mp.fac_s,
    )

    # SO4 Source (2.0x) - LIQUID
    # Couple to FeS2
    add_implicit_coupling(
        CROSS, RATES, "so4", "fes2", 2 * k.fes2_ox * c.o2, 2 * k.fes2_ox * c.o2 * c.fes2
    )
    # SO4_32
    add_implicit_coupling(
        CROSS,
        RATES,
        "so4_32",
        "fes2_32",
        k.fes2_ox * c.o2,
        k.fes2_ox * c.o2 * c.fes2_32,
    )


def equilibrate_fes_precipitation(c, k, mp, dt, RATES):
    """
    Instantaneous equilibrium 'Clip' for FeS precipitation.
    Solves (Fe - x)(S - x) = Ksp for x.
    """
    import numpy as np

    # 1. Get concentrations (Create explicit copies if you need them to stay static,
    # or just be careful with update order)
    fe = c.fe2_total.value * mp.f_diss
    hs = c.h2s.value  # Reference to live array

    # 2. Define Effective Ksp
    K_target = k.fes_sp * k.hplus  # Assuming k.hplus is available

    # 3. Identify Supersaturated Cells
    iap = fe * hs
    mask = iap > K_target

    if not np.any(mask):
        return

    # 4. Solve Quadratic
    A = 1.0
    B = -(fe[mask] + hs[mask])
    C = (fe[mask] * hs[mask]) - K_target

    delta = B**2 - 4.0 * A * C
    x_precip = (-B - np.sqrt(delta)) / 2.0

    # --- REPORTING LOGIC ---
    # Convert the 'jump' in concentration into a rate: mol/L/s
    # We use the same volume scaling as your matrix-based rates.
    rate_report = x_precip / dt

    # Update the diagnostic RATES dictionary
    # For H2S (Porewater units)
    RATES["h2s"][mask] -= rate_report

    # For Fe2_total (Bulk units)
    RATES["fe2_total"][mask] -= rate_report * mp.phi

    # For FeS (Solid units)
    RATES["fes"][mask] += rate_report * mp.fac_s
    # ---------------------------------------------------------
    # CRITICAL FIX: Calculate Isotope Mass Transfer FIRST
    # ---------------------------------------------------------
    if hasattr(c, "h2s_32"):
        # We must use hs[mask] HERE, before we subtract x_precip from it.
        # This gives us the fraction of the PRE-PRECIPITATION pool.

        # Optional: Add fractionation factor 'alpha' (e.g., 1.035 for faster 32S precip)
        # alpha = 1.0

        # Fraction of total H2S being removed
        frac_precip = x_precip / (hs[mask] + 1e-30)

        # Calculate the mass of 32S to move
        loss_32 = c.h2s_32.value[mask] * frac_precip  # * alpha

        # Update Isotope State Variables
        c.h2s_32.value[mask] -= loss_32
        c.fes_32.value[mask] += loss_32 * mp.fac_s

    # ---------------------------------------------------------
    # 5. Update Bulk State Variables (AFTER Isotope Calc)
    # ---------------------------------------------------------

    # Update H2S (Porewater) - This modifies 'hs' via reference!
    c.h2s.value[mask] -= x_precip

    # Update Fe2_total (Bulk)
    c.fe2_total.value[mask] -= x_precip * mp.phi

    # Update FeS (Solid)
    c.fes.value[mask] += x_precip * mp.fac_s

    # Diagnostic print (useful for debugging stiffness)
    # print(f"  [Equil] Cells: {np.sum(mask)} | Max precip: {np.max(x_precip):.2e}")

    return RATES


def equilibrate_fes_precipitation_old(c, k, mp):
    """
    Instantaneous equilibrium 'Clip' for FeS precipitation.
    Solves (Fe - x)(S - x) = Ksp for x.
    """
    import numpy as np

    # 1. Get concentrations (Porewater units)
    fe = (
        c.fe2_total.value * mp.f_diss
    )  # Assuming fe2_total is bulk, getting dissolved part
    hs = c.h2s.value

    # 2. Define Effective Ksp (Saturation target)
    # If your Ksp is defined as [Fe][H2S]/[H+] = K, then Target = K * [H+]
    # Adjust based on your specific K definition
    K_target = k.fes_sp * k.hplus

    # 3. Identify Supersaturated Cells
    # We only precipitate if IAP > K_target
    iap = fe * hs
    mask = iap > K_target

    if not np.any(mask):
        return  # Nothing to do

    # 4. Solve Quadratic: ax^2 + bx + c = 0
    # Equation: (Fe - x)(S - x) = K
    # x^2 - (Fe + S)x + (Fe*S - K) = 0

    A = 1.0
    B = -(fe[mask] + hs[mask])
    C = (fe[mask] * hs[mask]) - K_target

    # Quadratic Formula: x = (-B - sqrt(B^2 - 4AC)) / 2A
    # We choose the minus sign for the root because we want the smallest x
    # that satisfies the condition (we can't precipitate more than we have).
    delta = B**2 - 4.0 * A * C
    x_precip = (-B - np.sqrt(delta)) / 2.0

    # 5. Update the State Variables (Mass Transfer)

    # Remove x from dissolved phase
    # Note: If c.fe2_total is BULK, we remove x / f_diss (simplified view)
    # or better: remove 'x' from the porewater portion.

    # Update H2S (Porewater)
    c.h2s.value[mask] -= x_precip

    # Update Fe2_total (Bulk)
    # We removed 'x' moles/L_porewater.
    # Convert to Bulk removal: x * phi
    c.fe2_total.value[mask] -= x_precip * mp.phi

    # Add to FeS (Solid)
    # We added 'x' moles/L_porewater.
    # Convert to Solid concentration: x * (phi / (1-phi)) -> x * fac_s
    c.fes.value[mask] += x_precip * mp.fac_s

    # 6. Optional: Isotope Tracking (32S)
    if hasattr(c, "h2s_32"):
        # The fraction of S that precipitated is x / S_total
        # Remove that same fraction from 32S
        frac_precip = x_precip / (hs[mask] + 1e-30)

        loss_32 = c.h2s_32.value[mask] * frac_precip

        c.h2s_32.value[mask] -= loss_32
        c.fes_32.value[mask] += loss_32 * mp.fac_s


#  print(f"  [Equilibration] Adjusted {np.sum(mask)} cells. Max precip: {np.max(x_precip):.2e}")
