"""Define the reactions."""

import numpy as np


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
    limiters["inhib_o2"] = eps / (c.o2.value + eps)

    # Sulfate Limiter (Implicit 1/[S+K] and Explicit [S]/[S+K])
    K_so4 = 0.2
    limiters["so4_implicit"] = 1.0 / (c.so4.value + K_so4)
    limiters["so4_32_implicit"] = 1.0 / (c.so4_32.value + K_so4)

    limiters["so4_explicit"] = c.so4.value / (c.so4.value + K_so4)
    limiters["so4_32_explicit"] = c.so4_32.value / (c.so4_32.value + K_so4)

    limiters["fe3_explicit"] = 1.0  # c.fe3 / (c.fe3 + 1e-3)
    limiters["fe3_implicit"] = 1.0  # 1.0 / (c.fe3 + 1e-3)

    K_alpha = 0.2
    limiters["alpha_explicit"] = c.so4.value / (c.so4.value + K_alpha)
    limiters["alpha_implicit"] = 1.0 / (c.so4.value + K_alpha)

    # H2S Alpha Limiter (prevents numerical issues at trace concentrations)
    limiters["h2s_alpha_explicit"] = c.h2s.value / (c.h2s.value + 0.05)

    # update k-values
    k.fe3_h2s = calculate_k_iron_reduction(c.fe3.value, c.h2s.value)

    # 3. RUN PROCESSES
    # ----------------
    # Each function updates LHS, RHS, and RATES in place

    sulfate_reduction(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    aerobic_respiration(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    iron_reduction_h2s(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    h2s_oxidation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    fe2_oxidation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    fes_oxidation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    pyrite_oxidation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    pyrite_formation_s0(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    pyrite_formation_fes_h2s(c, k, limiters, LHS, RHS, RATES, CROSS, mp)
    iron_sulfide_formation(c, k, limiters, LHS, RHS, RATES, CROSS, mp)

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


def add_explicit_source(RHS, RATES, species, rate):
    """Add a production term to the RHS vector.

    Add a production term to the RHS vector.
    RHS = -Rate (Standard library quirk for production)
    """
    RHS[species] = RHS[species] - rate
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


def sulfate_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate sulfate reduction.

    Reaction: 2 POC + 1 SO4 -> 1 H2S Ref: POC (k.poc_so4)
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # 1. Base Rate
    rate_explicit = k.poc_so4 * c.poc * c.so4 * lim["so4_implicit"] * lim["inhib_o2"]

    # 2. POC Sink (Ref Species) - SOLID
    coeff_poc = k.poc_so4 * lim["so4_explicit"] * lim["inhib_o2"] * fac_s
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, rate_explicit * fac_s)

    # 3. SO4 Sink -> Rate = 0.5 * Base - LIQUID
    coeff_so4 = k.poc_so4 * c.poc * lim["inhib_o2"] * lim["so4_implicit"] * 0.5
    add_implicit_sink(LHS, RATES, "so4", coeff_so4, rate_explicit * 0.5)

    # 4. H2S Source (Coupled to POC)
    # Rate = 0.5 * k * SO4 * POC
    # H2S is liquid, so scaling is phi. coeff passed to coupling is unscaled porewater coeff.
    # Coeff = 0.5 * k * SO4
    coeff_h2s_coupling = 0.5 * k.poc_so4 * c.so4 * lim["inhib_o2"]
    add_implicit_coupling(
        CROSS, RATES, "h2s", "poc", coeff_h2s_coupling, rate_explicit * 0.5
    )

    # isotopes
    alpha = 1.0 + (mp.msr_alpha - 1.0) * lim["alpha_explicit"]
    s_val = c.so4.value + 1e-12
    s32_val = c.so4_32.value + 1e-12
    f_32 = alpha / (s_val + (alpha - 1) * s32_val + 1e-30)
    coeff_so4_32 = f_32 * rate_explicit

    # sulfate 32
    add_implicit_sink(
        LHS, RATES, "so4_32", coeff_so4_32 * 0.5, coeff_so4_32 * c.so4_32 * 0.5
    )

    # H2S_32 Source (Coupled to SO4_32? Or POC?)
    # Rate_32 = 0.5 * (f_32 * k * POC * SO4).
    # Use f_32 * Rate_base.
    # Rate_32 depends on SO4_32 linearly?
    # f_32 approx alpha * SO4_32 / SO4.
    # So Rate_32 approx alpha * k * POC * SO4_32.
    # So we couple to SO4_32.
    coupling_coeff_32 = 0.5 * alpha * k.poc_so4 * c.poc * lim["inhib_o2"]
    add_implicit_coupling(
        CROSS,
        RATES,
        "h2s_32",
        "so4_32",
        coupling_coeff_32,
        coeff_so4_32 * c.so4_32 * 0.5,
    )


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


def iron_reduction_h2s(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define iron reduction by h2s.
    Assumption: Fe2+ behaves as a solid (sorbed).
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # Fe3 Sink - SOLID
    coeff_fe3 = k.fe3_h2s * c.h2s * fac_s
    add_implicit_sink(LHS, RATES, "fe3", coeff_fe3, coeff_fe3 * c.fe3)

    # H2S Sink (0.5x) - LIQUID
    coeff_h2s = 0.5 * k.fe3_h2s * c.fe3
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s_32)

    # Fe2+ Source (1.0x) - Liquid (User Changed Fe2 to Dissolved!)
    # Rate = k * Fe3 * H2S
    # Couple to Fe3 (Solid source)
    # Fe2 is liquid -> scaling phi.
    # Equation: d(phi*Fe2)/dt = phi * k * H2S * Fe3
    # diff_lib adds: coeff*phi * Fe3.
    # So coeff = k * H2S.
    # But wait, Fe3 is Solid. Does it naturally scale?
    # No, C_Fe3 is mole/L_bulk.
    # Rate expression k * Fe3 * H2S uses molar concentrations.
    # k is typically liters/mol/s.
    # If Fe2 is liquid, and Fe3 is solid (conc per bulk vol), we just use concentrations.
    # coeff = k * H2S.
    rate_fe2 = k.fe3_h2s * c.fe3 * c.h2s * lim["fe3_explicit"]
    add_implicit_coupling(
        CROSS,
        RATES,
        "fe2",
        "fe3",
        k.fe3_h2s * c.h2s * lim["fe3_explicit"] * fac_s,
        rate_fe2 * fac_s,
    )

    # S0 Source (0.5x) - SOLID
    # Couple to Fe3
    s0_rate = 0.5 * k.fe3_h2s * c.fe3 * c.h2s
    # S0 is solid -> scaling 1-phi.
    # Equation d((1-phi)S0)/dt = phi * (0.5 * k * Fe3 * H2S).
    # coeff passed to diff_lib must satisfy: coeff * (1-phi) = phi * 0.5 * k * H2S.
    # coeff = (phi/(1-phi)) * 0.5 * k * H2S = fac_s * 0.5 * k * H2S.
    add_implicit_coupling(
        CROSS, RATES, "s0", "fe3", fac_s * 0.5 * k.fe3_h2s * c.h2s, s0_rate * fac_s
    )

    # S0_32 Couple to Fe3 (and ratio of H2S_32?)
    # Rate_32 roughly 0.5 * k * Fe3 * H2S_32.
    # So couple to H2S_32?
    # Fe3 consumption doesn't distinguish isotopes, but S0 comes from H2S.
    # So S0_32 comes from H2S_32.
    # Rate_32 = 0.5 * k * Fe3 * H2S_32.
    # Couple to H2S_32.
    # coeff = fac_s * 0.5 * k * Fe3.
    add_implicit_coupling(
        CROSS,
        RATES,
        "s0_32",
        "h2s_32",
        fac_s * 0.5 * k.fe3_h2s * c.fe3,
        0.5 * k.fe3_h2s * c.fe3 * c.h2s_32 * fac_s,
    )


def fe2_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 4 FeS2+ O2 -> 1 Fe3OOH"""
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    rate_base = k.fe2_ox * c.fe2 * c.o2

    # Fe2+ Sink - Solid
    coeff_fe2 = k.fe2_ox * c.o2
    add_implicit_sink(LHS, RATES, "fe2", coeff_fe2 * fac_s, rate_base * fac_s)

    # O2 Sink (1/4) - LIQUID
    coeff_o2 = 0.25 * k.fe2_ox * c.fe2
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, rate_base * 0.25)

    # Fe3 Source (1.0x) - SOLID
    # Couple to Fe2 (Liquid)
    # Fe3 is solid (fac_s scaling needed).
    # d((1-phi)Fe3)/dt = phi * k * Fe2 * O2.
    # coeff * (1-phi) = phi * k * O2.
    # coeff = fac_s * k * O2.
    add_implicit_coupling(
        CROSS, RATES, "fe3", "fe2", fac_s * k.fe2_ox * c.o2, rate_base * fac_s
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


def h2s_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 H2S + 0.5 O2 -> 1 S0"""
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # H2S Sink - LIQUID
    coeff_h2s = k.h2s_ox * c.o2
    alpha = 1.0 + (mp.h2s_ox_alpha - 1.0) * lim["h2s_alpha_explicit"]
    s_val = c.h2s.value + 1e-12
    s32_val = c.h2s_32.value + 1e-12
    f_32 = alpha / (s_val + (alpha - 1) * s32_val + 1e-30)

    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)

    coeff_h2s_32 = f_32 * coeff_h2s * c.h2s / (c.h2s_32 + 1e-30)
    # Approx f_32 * k * O2 * (H2S/H2S_32) ?? No.
    # Rate_32 = alpha * k * O2 * H2S_32.
    # Coeff_32 = alpha * k * O2.
    # Use that directly for stability.
    coeff_h2s_32_stable = alpha * k.h2s_ox * c.o2
    add_implicit_sink(
        LHS, RATES, "h2s_32", coeff_h2s_32_stable, coeff_h2s_32_stable * c.h2s_32
    )

    # O2 Sink (0.5x) - LIQUID
    coeff_o2 = 0.5 * k.h2s_ox * c.h2s
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2)

    # S0 Source (1.0x) - SOLID
    # Couple to H2S
    # Target Solid -> fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "s0",
        "h2s",
        fac_s * k.h2s_ox * c.o2,
        k.h2s_ox * c.o2 * c.h2s * fac_s,
    )
    # S0_32 couple to H2S_32
    add_implicit_coupling(
        CROSS,
        RATES,
        "s0_32",
        "h2s_32",
        fac_s * coeff_h2s_32_stable,
        coeff_h2s_32_stable * c.h2s_32 * fac_s,
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
    Method: Switch Kinetic Model with Full Linearization (Taylor Expansion)
    """
    # 1. Porosity correction
    phi = mp.phi
    fac_s = phi / (1.0 - phi)  # Converts Porewater Rate -> Solid Rate

    # 2. Saturation State
    omega_den = mp.hplus * k.isp_fes + 1e-30
    omega = (c.fe2 * c.h2s) / omega_den

    # 3. Switches
    is_supersat = omega > 1.0
    is_undersat = omega <= 1.0

    # =======================================================================
    # PRECIPITATION (Omega > 1)
    # Rate = k_ISP * (Omega - 1)  =  k_ISP * Omega  -  k_ISP
    # =======================================================================
    k_isp = k.fe2_h2s

    # A. Implicit Sink Term ( k_ISP * Omega )
    precip_coeff_fe2 = (k_isp * c.h2s / omega_den) * is_supersat
    precip_coeff_h2s = (k_isp * c.fe2 / omega_den) * is_supersat

    # B. Explicit Source Correction ( -k_ISP )
    precip_const_correction = k_isp * is_supersat

    # =======================================================================
    # DISSOLUTION (Omega < 1)
    # Rate = k_ISD * [FeS] * (1 - Omega)  =  k_ISD*[FeS]  -  k_ISD*[FeS]*Omega
    # =======================================================================
    k_isd = k.isd_fes
    conv = 1.0 / fac_s

    # A. Explicit Source Term ( k_ISD * [FeS] )
    diss_source_max = k_isd * c.fes * conv * is_undersat

    # B. Implicit Damping Term ( - k_ISD * [FeS] * Omega )
    diss_damping_base = (k_isd * c.fes * conv) / omega_den * is_undersat

    diss_coeff_fe2 = diss_damping_base * c.h2s
    diss_coeff_h2s = diss_damping_base * c.fe2

    # =======================================================================
    # APPLY TERMS (Linearized)
    # =======================================================================

    # --- Fe2 (Liquid) ---
    # Sink side: Precipitation + Dissolution Damping
    add_implicit_sink(LHS, RATES, "fe2", precip_coeff_fe2 + diss_coeff_fe2, 0.0)
    # Source side: Dissolution Max + Precip Correction
    add_explicit_source(RHS, RATES, "fe2", diss_source_max + precip_const_correction)

    # --- H2S (Liquid) ---
    add_implicit_sink(LHS, RATES, "h2s", precip_coeff_h2s + diss_coeff_h2s, 0.0)
    add_explicit_source(RHS, RATES, "h2s", diss_source_max + precip_const_correction)

    # --- FeS (Solid) ---
    term_precip = precip_coeff_fe2 * c.fe2 - precip_const_correction
    term_diss = diss_source_max - diss_coeff_fe2 * c.fe2

    # 1. The "+ k * Omega" part (Source for FeS, Sink for Fe2)
    # Coupled Source from Fe2 to FeS.
    add_implicit_coupling(
        CROSS, RATES, "fes", "fe2", precip_coeff_fe2 * fac_s, term_precip * fac_s
    )

    # 2. The "- k * 1" part (Correction)
    add_explicit_source(RHS, RATES, "fes", -precip_const_correction * fac_s)

    # Sink: Dissolution
    diss_coeff_solid = k_isd * (1.0 - omega) * is_undersat
    add_implicit_sink(LHS, RATES, "fes", diss_coeff_solid, term_diss * fac_s)

    # --- ISOTOPES (32S) ---
    if hasattr(c, "h2s_32"):
        # Fe2 is the partner.
        add_implicit_sink(LHS, RATES, "h2s_32", precip_coeff_h2s + diss_coeff_h2s, 0.0)

        # H2S_32 Source (Liquid)
        ratio_liquid = c.h2s_32 / (c.h2s + 1e-20)
        ratio_solid = c.fes_32 / (c.fes + 1e-20)

        src_precip_32 = precip_const_correction * ratio_liquid
        src_diss_32 = diss_source_max * ratio_solid

        add_explicit_source(RHS, RATES, "h2s_32", src_precip_32 + src_diss_32)

        # FeS_32 Source (Solid)
        coeff_precip_32 = (k_isp / (omega_den + 1e-30)) * c.fe2 * is_supersat
        rate_precip_32 = coeff_precip_32 * c.h2s_32 * fac_s

        add_implicit_coupling(
            CROSS, RATES, "fes_32", "h2s_32", coeff_precip_32 * fac_s, rate_precip_32
        )

        add_explicit_source(
            RHS, RATES, "fes_32", -precip_const_correction * ratio_liquid * fac_s
        )

        add_implicit_sink(
            LHS, RATES, "fes_32", diss_coeff_solid, term_diss * ratio_solid * fac_s
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
