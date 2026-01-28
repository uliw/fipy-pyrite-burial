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
    # iron_sulfide_formation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
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

    return f


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
    RATES[species] -= rate


def add_explicit_source_old(RHS, RATES, species, rate):
    """Add a production term to the RHS vector.

    Add a production term to the RHS vector.
    RHS = -Rate (Standard library quirk for production)
    """
    # RHS[species] = RHS[species] - rate
    RHS[species] -= rate
    RATES[species] += rate  # reporting only


def add_explicit_source(RHS, RATES, species, rate):
    """Add a production term to the RHS vector.

    Add a production term to the RHS vector.
    RHS = -Rate (Standard library quirk for production)
    """
    RHS[species] = RHS[species] + rate
    RATES[species] += rate


def add_implicit_coupling(CROSS, RATES, target_species, source_species, coeff, rate):
    """
    Add a coupled source term.

    If d[Target]/dt = +coeff * [Source]
    Then we add `ImplicitSourceTerm(coeff=coeff, var=Source)` to Target's equation.

    CROSS[target].append( (source, coeff) )
    """
    CROSS[target_species].append((source_species, coeff))
    # Note: Rates are accumulating scalar values for reporting, usually calculated explicitly before calling
    RATES[target_species] += rate


# =============================================================================
# PROCESS FUNCTIONS (The Biogeochemistry)
# =============================================================================
def aerobic_respiration(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define POC consumption by aerobic respiration."""
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    rate_base = k.poc_o2 * c.poc * c.o2

    # POC Sink - SOLID
    coeff_poc = k.poc_o2 * c.o2 * fac_s
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, rate_base * fac_s)

    # O2 Sink (1.27x) - LIQUID
    coeff_o2 = 1.27 * k.poc_o2 * c.poc
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, 1.27 * rate_base)
    # No produced species here (CO2 ignored)


def sulfate_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate sulfate reduction.

    Reaction: 2 POC + 1 SO4 -> 1 H2S Ref: POC (k.poc_so4)
    """
    fac_s = mp.phi / (1.0 - mp.phi)  # for solid species

    # 1. Base Rate
    poc_rate = k.poc_so4 * c.poc * c.so4 * lim["so4_implicit"] * lim["inhib_o2"]
    so4_rate = poc_rate * 0.5

    # 2. POC Sink (Ref Species) - SOLID
    coeff_poc = k.poc_so4 * lim["so4_explicit"] * lim["inhib_o2"] * fac_s
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, poc_rate * fac_s)

    # 3. SO4 Sink -> Rate = 0.5 * Base - LIQUID
    coeff_so4 = k.poc_so4 * c.poc * lim["inhib_o2"] * lim["so4_implicit"] * 0.5
    add_implicit_sink(LHS, RATES, "so4", coeff_so4, so4_rate)

    # 4. H2S Source as coupled to sulfate reduction
    add_implicit_coupling(
        CROSS,
        RATES,
        "h2s",  # species that is produced
        "so4",  # source species
        coeff_so4,
        so4_rate,
    )

    # isotopes
    if hasattr(c, "so4_32"):
        alpha = 1.0 + (mp.msr_alpha - 1.0) * lim["alpha_explicit"]
        s_val = c.so4 + 1e-12
        s32_val = c.so4_32 + 1e-12
        f_32 = alpha / (s_val + (alpha - 1) * s32_val + 1e-30)
        coeff_so4_32 = f_32 * so4_rate

        # sulfate 32
        add_implicit_sink(LHS, RATES, "so4_32", coeff_so4_32, coeff_so4_32 * c.so4_32)

        # note this coupled recation only adds on h2s_32, so it does not affect so4_32!
        add_implicit_coupling(
            CROSS,
            RATES,
            "h2s_32",  # species that is produced
            "so4_32",  # source species
            coeff_so4_32,  # implicit coeff for sink
            coeff_so4_32 * c.so4_32,  # explicit rate for reporting
        )


def h2s_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 H2S + 0.5 O2 -> 1 S0"""
    fac_s = mp.phi / (1.0 - mp.phi)

    # H2S Sink - LIQUID
    # Ref: H2S
    coeff_h2s = k.h2s_ox * c.o2
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)

    # O2 Sink (0.5x) - LIQUID
    coeff_o2 = 0.5 * k.h2s_ox * c.h2s
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2)

    # S0 Source (1.0x) - SOLID, Couple to H2S
    add_implicit_coupling(
        CROSS,
        RATES,
        "s0",  # species that is produced
        "h2s",  # source species
        coeff_h2s * fac_s,  # implicit coeff for sink
        coeff_h2s * c.h2s * fac_s,  # explicit rate for reporting
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

        add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s_32, coeff_h2s_32 * c.h2s_32)

        # S0_32 coupled to H2S_32
        add_implicit_coupling(
            CROSS,
            RATES,
            "s0_32",  # species that is produced
            "h2s_32",  # source species
            coeff_h2s_32 * fac_s,  # implicit coeff for sink
            coeff_h2s_32 * c.h2s_32 * fac_s,  # explicit rate for reporting
        )


def iron_reduction_h2s_lumped(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define iron reduction by h2s.
    1 Fe3 + 0.5 H2S -> 1 Fe2 + 0.5 S0
    Fe2+ is a liquid!
    """
    fac_s = mp.phi / (1.0 - mp.phi)
    k.fe3_h2s = calculate_k_iron_reduction(c.fe3, c.h2s)

    # 1. Fe3 Sink - SOLID (Linear)
    # Rate = k * [H2S] * [Fe3]
    coeff_fe3 = k.fe3_h2s * c.h2s * fac_s
    add_implicit_sink(LHS, RATES, "fe3", coeff_fe3, coeff_fe3 * c.fe3)

    # 2. H2S Sink (0.5x) - LIQUID (Linear)
    coeff_h2s = k.fe3_h2s * c.fe3 * 0.5
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)

    if hasattr(c, "h2s_32"):
        add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s_32)

    # 3. Fe2+ Source (1.0x) - Liquid (Coupled to Fe3)
    # MUST BE LINEAR to match Fe3 Sink
    # Rate = k * [H2S] * [Fe3]
    # Coeff = k * [H2S]

    coeff_coupling_fe2 = k.fe3_h2s * c.h2s

    add_implicit_coupling(
        CROSS,
        RATES,
        "fe2_total",  # target
        "fe3",  # source
        coeff_coupling_fe2,  # implicit coefficient
        coeff_coupling_fe2 * c.fe3,  # explicit rate
    )

    # 4. Elemental sulfur - Solid (Coupled to H2S)
    # Rate = 0.5 * k * [Fe3] * [H2S] * fac_s
    # Matches H2S Sink stoichiometry and kinetics exactly
    add_implicit_coupling(
        CROSS,
        RATES,
        "s0",
        "h2s",
        coeff_h2s * fac_s,
        coeff_h2s * c.h2s * fac_s,
    )

    if hasattr(c, "h2s_32"):
        # Elemental sulfur 32S - Solid
        add_implicit_coupling(
            CROSS,
            RATES,
            "s0_32",
            "h2s_32",
            coeff_h2s * fac_s,
            coeff_h2s * c.h2s_32 * fac_s,
        )


def iron_reduction_h2s(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define iron reduction by h2s.
    1 Fe3 + 0.5 H2S -> 1 Fe2 + 0.5 S0
    Fe2+ is a liquid!
    """
    fac_s = mp.phi / (1.0 - mp.phi)
    k.fe3_h2s = calculate_k_iron_reduction(c.fe3, c.h2s)

    # 1. Fe3 Sink - SOLID (Linear)
    # Rate = k * [H2S] * [Fe3]
    coeff_fe3 = k.fe3_h2s * c.h2s * fac_s
    add_implicit_sink(LHS, RATES, "fe3", coeff_fe3, coeff_fe3 * c.fe3)

    # 2. H2S Sink (0.5x) - LIQUID (Linear)
    coeff_h2s = k.fe3_h2s * c.fe3 * 0.5
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)

    if hasattr(c, "h2s_32"):
        add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s_32)

    # 3. Fe2+ Source (1.0x) - Liquid (Coupled to Fe3)
    # MUST BE LINEAR to match Fe3 Sink
    # Rate = k * [H2S] * [Fe3]
    # Coeff = k * [H2S]

    coeff_coupling_fe2 = k.fe3_h2s * c.h2s

    add_implicit_coupling(
        CROSS,
        RATES,
        "fe2",  # target
        "fe3",  # source
        coeff_coupling_fe2,  # implicit coefficient
        coeff_coupling_fe2 * c.fe3,  # explicit rate
    )

    # 4. Elemental sulfur - Solid (Coupled to H2S)
    # Rate = 0.5 * k * [Fe3] * [H2S] * fac_s
    # Matches H2S Sink stoichiometry and kinetics exactly
    add_implicit_coupling(
        CROSS,
        RATES,
        "s0",
        "h2s",
        coeff_h2s * fac_s,
        coeff_h2s * c.h2s * fac_s,
    )

    if hasattr(c, "h2s_32"):
        # Elemental sulfur 32S - Solid
        add_implicit_coupling(
            CROSS,
            RATES,
            "s0_32",
            "h2s_32",
            coeff_h2s * fac_s,
            coeff_h2s * c.h2s_32 * fac_s,
        )


def fe2_adsoption_lumped(c, k, lim, LHS, RHS, RATES, Cross, mp):
    """
    Handle Iron Partitioning algebraically.
    Instead of calculating rates, we calculate fractions.

    System State: 'fe_total' is the primary variable.
    fe2 (liquid) and fe2_p (solid) are derived helper views.
    """
    phi = mp.phi
    K_ads = k.fe2_p_eq  # 696

    # Check Units of K_ads!
    # If K_ads is dimensionless (Conc_solid_vol / Conc_liquid_vol):
    #   Capacity = phi + (1-phi)*K_ads
    # If K_ads is (Conc_solid_mass / Conc_liquid_vol) [L/kg]:
    #   Capacity = phi + (1-phi)*rho*K_ads

    # Assuming K_ads is dimensionless (based on previous fac_s logic):
    R_factor = phi + (1.0 - phi) * K_ads

    # Calculate Fractions
    c.f_diss = phi / R_factor
    c.f_sorb = (1.0 - phi) * K_ads / R_factor

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
        c.fe2.setValue(c.fe2_total * c.f_diss)
        c.fe2_p.setValue(c.fe2_total * c.f_sorb)


def fe2_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Reaction: 4 FeS2+ O2 -> 1 Fe3OOH
    Note: Fe2_total tracks Fe2 liquid and sorbed. We need to use
    the respective fractions c.f_diss and c.f_sorb
    """
    fac_s = mp.phi / (1.0 - mp.phi)

    # mp.f_diss = dissolved fraction of fe2_total
    rate_base = k.fe2_ox * c.fe2_total * c.o2

    # Fe2+ Sink - Liquid
    coeff_fe2 = k.fe2_ox * c.o2
    add_implicit_sink(LHS, RATES, "fe2_total", coeff_fe2, rate_base)

    # O2 Sink (1/4) - LIQUID
    coeff_o2 = k.fe2_ox * c.fe2_total * 0.25
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, rate_base * 0.25)

    # Fe3 Source (1.0x) - SOLID
    # Couple to Fe2 (Liquid)
    # Fe3 is solid (fac_s scaling needed).
    # d((1-phi)Fe3)/dt = phi * k * Fe2 * O2.
    # coeff * (1-phi) = phi * k * O2.
    # coeff = fac_s * k * O2.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fe3",  # product
        "fe2_total",  # source
        k.fe2_ox * c.o2 * fac_s,  # coefficient
        rate_base * fac_s,  # rate for reporting
    )


def fes_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 2.25 O2 -> 1 Fe3 + 1 SO4"""
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    rate_base = k.fes_ox * c.fes * c.o2
    rate_base_32 = k.fes_ox * c.fes_32 * c.o2

    # FeS Sink - SOLID
    coeff_fes = k.fes_ox * c.o2 * fac_s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, rate_base * fac_s)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, rate_base_32 * fac_s)

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
    # If we use fac_sLogic?
    # Let's assume consistent k units.
    # Reactions involving solids are usually scaled by (1-phi) in the solver logic for "divided" equations?
    # No, 'divided' means we modeled C, not (1-phi)C.
    # But `diff_lib` multiplies by `scaling`.
    # So `dC/dt = ... + k*C`.
    # So `coeff = k * O2`.
    # Wait, in the sink, we used `coeff_fes * fac_s`.
    # Why? `diagenetic_reactions` doc says "For Solid Species: R_divided = R_base * (phi/(1-phi))".
    # This implies R_base is porewater rate.
    # If we couple FeS (Solid) -> Fe3 (Solid), and both use porewater rate base?
    # Then we need `fac_s`.
    add_implicit_coupling(
        CROSS, RATES, "fe3", "fes", k.fes_ox * c.o2 * fac_s, rate_base * fac_s
    )

    # SO4 Source (1.0x) - LIQUID
    # Couple to FeS.
    # Target Liquid. No fac_s.
    add_implicit_coupling(CROSS, RATES, "so4", "fes", k.fes_ox * c.o2, rate_base)
    add_implicit_coupling(
        CROSS, RATES, "so4_32", "fes_32", k.fes_ox * c.o2, rate_base_32
    )


def pyrite_formation_s0(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 1 S0 -> 1 FeS2"""
    phi = mp.phi
    fac_s = phi / (1.0 - phi)  # Should we use fac_s?
    # Original comment said "Solid-Solid reactions do not need phi/(1-phi)" but then used explicit source (unscaled).
    # If we use implicit sink with `fac_s` for S0 and FeS (as in original code),
    # then we must use `fac_s` for FeS2 source too to maintain mass balance
    # (destroy 1 mol S0/FeS -> make 1 mol FeS2, relative to bulk).

    # S0 Sink - SOLID
    coeff_s0 = k.fes_s0 * c.fes * fac_s
    add_implicit_sink(LHS, RATES, "s0", coeff_s0, coeff_s0 * c.s0)
    add_implicit_sink(LHS, RATES, "s0_32", coeff_s0, coeff_s0 * c.s0_32)

    # FeS Sink (1.0x) - SOLID
    coeff_fes = k.fes_s0 * c.s0 * fac_s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, coeff_fes * c.fes)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, coeff_fes * c.fes_32)

    # FeS2 Source (1.0x) - SOLID
    # Couple to FeS?
    # Rate = k * FeS * S0.
    # Couple to FeS: coeff = k * S0 * fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2",
        "fes",
        k.fes_s0 * c.s0 * fac_s,
        k.fes_s0 * c.s0 * c.fes * fac_s,
    )

    # FeS2_32 Source
    # Sum of S0_32 and FeS_32 ?
    # FeS2 contains 2 sulfurs.
    # Rate FeS2_32 is purely tracking S32 mass.
    # S32 comes from S0_32 and FeS_32.
    # Couple to both!
    # Term 1: from S0_32. Rate = k * FeS * S0_32. Coeff = k * FeS * fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "s0_32",
        k.fes_s0 * c.fes * fac_s,
        k.fes_s0 * c.fes * c.s0_32 * fac_s,
    )
    # Term 2: from FeS_32. Rate = k * S0 * FeS_32. Coeff = k * S0 * fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "fes_32",
        k.fes_s0 * c.s0 * fac_s,
        k.fes_s0 * c.s0 * c.fes_32 * fac_s,
    )


def pyrite_formation_fes_h2s(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 1 H2S -> 1 FeS2"""
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # FeS Sink - SOLID
    coeff_fes = k.fes_h2s * c.h2s * fac_s
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
        k.fes_h2s * c.h2s * fac_s,
        k.fes_h2s * c.h2s * c.fes * fac_s,
    )

    # FeS2_32 Source
    # From FeS_32 and H2S_32.
    # Couple to FeS_32
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "fes_32",
        k.fes_h2s * c.h2s * fac_s,
        k.fes_h2s * c.h2s * c.fes_32 * fac_s,
    )
    # Couple to H2S_32
    # Rate = k * FeS * H2S_32.
    # Target Solid, Source Liquid. Coeff Needs fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "h2s_32",
        fac_s * k.fes_h2s * c.fes,
        fac_s * k.fes_h2s * c.fes * c.h2s_32,
    )


def apply_rate_limiter(rate, var, fraction=0.5, eps=1e-12):
    """Limit rate so it doesn't consume more than a fraction of available var."""
    val = var.value if hasattr(var, "value") else var
    max_rate = val * fraction / 1.0  # Normalized dt=1 for steady state sweep
    return np.minimum(rate, np.maximum(max_rate, 0.0))


def iron_sulfide_formation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Reaction: Fe2 + H2S <-> FeS
    Method: Switch Kinetic Model with Full Linearization
    """
    # 1. Porosity correction
    phi = mp.phi
    fac_s = phi / (1.0 - phi)  # Converts Porewater Rate -> Solid Rate

    # 2. Clamping (Prevent negative concentrations in rate laws)
    fe2_pos = (c.fe2 + abs(c.fe2)) * 0.5
    h2s_pos = (c.h2s + abs(c.h2s)) * 0.5
    fes_pos = (c.fes + abs(c.fes)) * 0.5

    # 3. Saturation State
    # Omega = [Fe][H2S] / (H * Ksp)
    omega_den = mp.hplus * k.isp_fes + 1e-30
    omega = (fe2_pos * h2s_pos) / omega_den

    # 4. Switches
    is_supersat = omega > 1.0
    is_undersat = omega <= 1.0

    # =======================================================================
    # PRECIPITATION (Omega > 1)
    # Rate = k_ISP * (Omega - 1)  =  k_ISP * Omega  -  k_ISP
    # =======================================================================
    k_isp = k.fe2_h2s

    # A. Implicit Sink Coefficients ( k_ISP * Omega / [Species] )
    precip_coeff_fe2 = (k_isp * h2s_pos / omega_den) * is_supersat
    precip_coeff_h2s = (k_isp * fe2_pos / omega_den) * is_supersat

    # B. Explicit Source Correction ( k_ISP )
    precip_const_correction = k_isp * is_supersat

    # =======================================================================
    # DISSOLUTION (Omega < 1)
    # Rate = k_ISD * [FeS] * (1 - Omega)
    # =======================================================================
    k_isd = k.isd_fes
    conv = 1.0 / fac_s

    # A. Explicit Source Term ( k_ISD * [FeS] )
    diss_source_max = k_isd * fes_pos * conv * is_undersat

    # B. Implicit Damping Term ( - k_ISD * [FeS] * Omega )
    diss_damping_base = (k_isd * fes_pos * conv) / omega_den * is_undersat
    diss_coeff_fe2 = diss_damping_base * h2s_pos
    diss_coeff_h2s = diss_damping_base * fe2_pos

    # =======================================================================
    # RATE REPORTING (Net Rates for Plots)
    # =======================================================================
    # Calculate the exact NET rates here to avoid double-counting corrections
    # Rate_Liquid = (Precipitation - Dissolution)
    #             = (k_isp * Omega - k_isp) - (k_isd * FeS * (1-Omega))

    rate_precip_net = precip_coeff_fe2 * fe2_pos - precip_const_correction
    rate_diss_net = diss_source_max - diss_coeff_fe2 * fe2_pos

    # Net Consumption of Fe2/H2S (Positive = Consumption)
    net_consumption_liquid = rate_precip_net - rate_diss_net

    # =======================================================================
    # APPLY TERMS TO SOLVER
    # =======================================================================

    # --- Fe2 (Liquid) ---
    # Sink: Precip (+ Damping)
    # We pass 0.0 for rate because we update RATES manually at the end
    add_implicit_sink(LHS, RATES, "fe2", precip_coeff_fe2 + diss_coeff_fe2, 0.0)

    # Source: Dissolution Max + Precip Correction
    # Target Equation: dC/dt = -Rate = -(Precip - Diss) = Diss - Precip
    # Diss term: +diss_source_max
    # Precip term: -(-k_isp) = +k_isp
    # We use 'add_explicit_source' which adds to RHS.
    add_explicit_source(RHS, RATES, "fe2", diss_source_max + precip_const_correction)

    # Manual Rate Update
    RATES["fe2"] -= net_consumption_liquid

    # --- H2S (Liquid) ---
    add_implicit_sink(LHS, RATES, "h2s", precip_coeff_h2s + diss_coeff_h2s, 0.0)
    add_explicit_source(RHS, RATES, "h2s", diss_source_max + precip_const_correction)
    RATES["h2s"] -= net_consumption_liquid

    # --- FeS (Solid) ---
    # Target Equation: dC/dt = +Rate * fac_s

    # 1. Source from Fe2 (Precipitation)
    # Logic: Source = (k_isp * Omega) * fac_s
    # We implement this as an Explicit Source using the CURRENT Fe2 value
    # to avoid complex coupling structures if not supported.
    # If using segregated solver, this uses Fe2 from previous sweep.
    # Note: This is the critical "Thin Air" fix. We MUST multiply by fe2_pos.
    source_from_fe2 = precip_coeff_fe2 * fe2_pos * fac_s
    add_explicit_source(RHS, RATES, "fes", source_from_fe2)

    # 2. Correction Term (Precipitation)
    # Logic: Correction = -k_isp * fac_s
    add_explicit_source(RHS, RATES, "fes", -precip_const_correction * fac_s)

    # 3. Sink (Dissolution)
    # Logic: Sink = -k_isd * (1-Omega) * [FeS]
    diss_coeff_solid = k_isd * (1.0 - omega) * is_undersat
    add_implicit_sink(LHS, RATES, "fes", diss_coeff_solid, 0.0)

    # Manual Rate Update (Production - Consumption)
    RATES["fes"] += net_consumption_liquid * fac_s

    # --- ISOTOPES (32S) ---
    if hasattr(c, "h2s_32"):
        # 1. H2S_32 Sink
        h2s_inv = 1.0 / (h2s_pos + 1e-20)
        coeff_precip_32 = precip_coeff_h2s - k_isp * h2s_inv * is_supersat
        coeff_precip_32 = np.maximum(coeff_precip_32, 0.0)

        # Rate calc for reporting
        rate_precip_32 = coeff_precip_32 * c.h2s_32

        add_implicit_sink(LHS, RATES, "h2s_32", coeff_precip_32 + diss_coeff_h2s, 0.0)

        # 2. FeS_32 Source (From H2S_32)
        # Source = coeff * H2S_32 * fac_s
        add_explicit_source(RHS, RATES, "fes_32", rate_precip_32 * fac_s)

        # 3. Dissolution (Source for H2S_32 from FeS_32)
        # Ratio = fes_32 / fes
        ratio_solid = c.fes_32 / (c.fes + 1e-20)
        rate_diss_32 = (
            diss_source_max * ratio_solid
        )  # Note: diss_source_max already has conv factor

        add_explicit_source(RHS, RATES, "h2s_32", rate_diss_32)

        # 4. FeS_32 Sink
        add_implicit_sink(LHS, RATES, "fes_32", diss_coeff_solid, 0.0)

        # Update Rates
        net_32 = rate_precip_32 - (diss_coeff_solid * c.fes_32 / fac_s)  # approx
        RATES["h2s_32"] -= net_32
        RATES["fes_32"] += net_32 * fac_s


def iron_sulfide_formation_old(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Reaction: Fe2 + H2S <-> FeS
    Method: Switch Kinetic Model with Full Linearization (Taylor Expansion)
    """
    # 1. Porosity correction
    phi = mp.phi
    fac_s = phi / (1.0 - phi)  # Converts Porewater Rate -> Solid Rate

    # 2. Clamping
    # Ensure non-negative concentrations for rate calculations to prevent instability
    fe2_pos = (c.fe2 + abs(c.fe2)) * 0.5
    h2s_pos = (c.h2s + abs(c.h2s)) * 0.5
    fes_pos = (c.fes + abs(c.fes)) * 0.5

    # 3. Saturation State
    omega_den = mp.hplus * k.isp_fes + 1e-30
    omega = (fe2_pos * h2s_pos) / omega_den

    # 4. Switches
    is_supersat = omega > 1.0
    is_undersat = omega <= 1.0

    # =======================================================================
    # PRECIPITATION (Omega > 1)
    # Rate = k_ISP * (Omega - 1)  =  k_ISP * Omega  -  k_ISP
    # =======================================================================
    k_isp = k.fe2_h2s

    # A. Implicit Sink Term ( k_ISP * Omega )
    # Note: Coefficients for implicit solver can/should use clamped values to guide solving?
    # Actually, if we want Implicit Linearization: Rate ~ k * fe2 * h2s.
    # If we use fe2_pos in coefficient, we linearize around the clamped value.
    # This is safer.
    precip_coeff_fe2 = (k_isp * h2s_pos / omega_den) * is_supersat
    precip_coeff_h2s = (k_isp * fe2_pos / omega_den) * is_supersat

    # B. Explicit Source Correction ( -k_ISP )
    precip_const_correction = k_isp * is_supersat

    # =======================================================================
    # DISSOLUTION (Omega < 1)
    # Rate = k_ISD * [FeS] * (1 - Omega)  =  k_ISD*[FeS]  -  k_ISD*[FeS]*Omega
    # =======================================================================
    k_isd = k.isd_fes
    conv = 1.0 / fac_s

    # A. Explicit Source Term ( k_ISD * [FeS] )
    # CRITICAL: Use clamped FeS to prevent negative source feedback loop
    diss_source_max = k_isd * fes_pos * conv * is_undersat

    # B. Implicit Damping Term ( - k_ISD * [FeS] * Omega )
    diss_damping_base = (k_isd * fes_pos * conv) / omega_den * is_undersat

    diss_coeff_fe2 = diss_damping_base * c.h2s
    diss_coeff_h2s = diss_damping_base * c.fe2

    # =======================================================================
    # APPLY TERMS (Linearized)
    # =======================================================================

    # --- Fe2 (Liquid) ---
    # Sink side: Precipitation + Dissolution Damping
    add_implicit_sink(LHS, RATES, "fe2", precip_coeff_fe2 + diss_coeff_fe2, 0.0)
    # Source side: Dissolution Max + Precip Correction
    # Dissolution Max is Source (+k_dis). Pass -k_dis.
    # Precip Correction is Source (+k_isp). Pass -k_isp.
    add_explicit_source(RHS, RATES, "fe2", -diss_source_max - precip_const_correction)

    # --- H2S (Liquid) ---
    add_implicit_sink(LHS, RATES, "h2s", precip_coeff_h2s + diss_coeff_h2s, 0.0)
    add_explicit_source(RHS, RATES, "h2s", -diss_source_max - precip_const_correction)

    # --- FeS (Solid) ---
    term_precip = precip_coeff_fe2 * c.fe2 - precip_const_correction
    term_diss = diss_source_max - diss_coeff_fe2 * c.fe2

    # 1. The "+ k * Omega" part (Source for FeS, Sink for Fe2)
    # Coupled Source from Fe2 to FeS.
    add_implicit_coupling(
        CROSS, RATES, "fes", "fe2", precip_coeff_fe2 * fac_s, term_precip * fac_s
    )

    # 2. The "- k * 1" part (Correction)
    # Correction is Sink (-k_isp). Pass +k_isp.
    add_explicit_source(RHS, RATES, "fes", precip_const_correction * fac_s)

    # Sink: Dissolution
    diss_coeff_solid = k_isd * (1.0 - omega) * is_undersat
    add_implicit_sink(LHS, RATES, "fes", diss_coeff_solid, term_diss * fac_s)

    # --- ISOTOPES (32S) ---
    if hasattr(c, "h2s_32"):
        # I. PRECIPITATION (H2S_32 -> FeS_32)
        # Rate_32 = Rate_Bulk * (H2S_32 / H2S) (assuming alpha=1)
        # Rate_Bulk = k_ISP * ( Fe2 * H2S / Den - 1 )
        # Rate_32 = k_ISP * ( Fe2 / Den - 1/H2S ) * H2S_32
        # Implicit Coeff = k_ISP * ( Fe2 / Den - 1/H2S )

        # Stability: 1/H2S can be large. However, we only precipitate if Omega > 1.
        # Omega = Fe2 * H2S / Den. So H2S > Den / Fe2.
        # So 1/H2S < Fe2 / Den.
        # So (Fe2 / Den - 1/H2S) should be positive and bounded.

        # 1. H2S_32 Sink (Liquid)
        h2s_inv = 1.0 / (h2s_pos + 1e-20)
        # coeff_precip_32 = k_isp * (c.fe2 / omega_den - h2s_inv) * is_supersat
        # Use existing calculation parts for efficiency/consistency
        coeff_precip_32 = precip_coeff_h2s - k_isp * h2s_inv * is_supersat

        # Clip to 0 to avoid numerical noise if Omega ~ 1
        coeff_precip_32 = np.maximum(coeff_precip_32, 0.0)

        add_implicit_sink(LHS, RATES, "h2s_32", coeff_precip_32 + diss_coeff_h2s, 0.0)

        # 2. FeS_32 Source (Solid)
        # Coupled Source from H2S_32 to FeS_32
        rate_precip_32 = coeff_precip_32 * c.h2s_32 * fac_s
        add_implicit_coupling(
            CROSS, RATES, "fes_32", "h2s_32", coeff_precip_32 * fac_s, rate_precip_32
        )

        # H2S_32 Source Correction: + k_isp/H2S * H2S_32 ??
        # Rate_32 = k_isp * Omega * H2S_32/H2S - k_isp * H2S_32/H2S.
        # Implicit part (Coeff*H2S_32) = k_isp * Omega * H2S_32/H2S - k_isp * H2S_32/H2S.
        # Wait.
        # Coeff used: k_isp * fe2 / den - k_isp / h2s.
        # Coeff * H2S_32 = k_isp * Omega/h2s * H2S_32 * fe2/fe2 ?? No.
        # Coeff = k_isp * Omega / H2S - k_isp / H2S.
        # So Coeff*H2S_32 = k_isp * Omega * (H2S_32/H2S) - k_isp * (H2S_32/H2S).
        # This matches Rate exactly!
        # So Normalized Rate_32 is fully covered by Implicit Term.
        # Is there a correction term needed?
        # Only if linearization error vs clamped values?
        # No explicit correction needed for Isotopes if fully implicit?

        # II. DISSOLUTION (FeS_32 -> H2S_32)
        # Rate_32 = Rate_Diss_Bulk * (FeS_32 / FeS)
        # Rate_Diss_Bulk = k_ISD * FeS * (1 - Omega)
        # Rate_32 = k_ISD * (1 - Omega) * FeS_32

        # 1. H2S_32 Source (Liquid)
        # Coupled Source from FeS_32 to H2S_32
        coeff_diss_32_coupling = k_isd * (1.0 - omega) * is_undersat * conv
        rate_diss_32 = coeff_diss_32_coupling * c.fes_32

        add_implicit_coupling(
            CROSS, RATES, "h2s_32", "fes_32", coeff_diss_32_coupling, rate_diss_32
        )

        # 2. FeS_32 Sink (Solid)
        # Implemented via `diss_coeff_solid` which is `k_isd * (1-Omega)`.
        term_diss_32_est = diss_coeff_solid * c.fes_32  # Estimate for reporting?
        add_implicit_sink(
            LHS, RATES, "fes_32", diss_coeff_solid, term_diss_32_est * fac_s
        )


def pyrite_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Reaction: 1 FeS2 + 3.5 O2 -> 1 Fe3 + 2 SO4
    Ref: FeS2 (k.fes2_ox)
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # FeS2 Sink - SOLID
    coeff_fes2 = k.fes2_ox * c.o2 * fac_s
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
        k.fes2_ox * c.o2 * fac_s,
        k.fes2_ox * c.o2 * c.fes2 * fac_s,
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
