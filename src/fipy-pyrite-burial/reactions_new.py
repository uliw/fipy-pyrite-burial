"""Define diagentic reactions."""


def diagenetic_reactions(mp, c, k, f):
    """Define diagenetic reactions.

    Main orchestrator for diagenetic reactions.
    Calculates limiters, initializes matrices, and calls specific process functions.
    """
    import numpy as np
    from diff_lib import calculate_k_iron_reduction

    # 1. SETUP & INITIALIZATION
    # -------------------------
    species_list = list(c.keys())

    # Accumulators (The State)
    # LHS: Matrix Diagonal (Implicit Sinks)
    # RHS: Vector (Explicit Sources)
    # RATES: For tracking/reporting

    # FIX: Initialize LHS with 0.0 * Identity to ensure it is a DiscretizedScalar object
    # Or just 0.0 as we will add to it.
    LHS = {s: 0.0 for s in species_list}

    RHS = {s: np.zeros_like(c.so4) for s in species_list}
    RATES = {s: np.zeros_like(c.so4) for s in species_list}

    # 2. CALCULATE LIMITERS
    # ---------------------
    eps = mp.eps
    limiters = {}

    # O2 Inhibition (1.0 -> 0.0)
    limiters["inhib_o2"] = eps / (c.o2.value + eps)

    # Sulfate Limiter (Implicit 1/[S+K] and Explicit [S]/[S+K])
    K_so4 = 0.1
    limiters["so4_implicit"] = 1.0 / (c.so4.value + K_so4)
    limiters["so4_32_implicit"] = 1.0 / (c.so4_32.value + K_so4)

    limiters["so4_explicit"] = c.so4.value / (c.so4.value + K_so4)
    limiters["so4_32_explicit"] = c.so4_32.value / (c.so4_32.value + K_so4)

    limiters["fe3_explicit"] = 1.0  # c.fe3 / (c.fe3 + 1e-3)
    limiters["fe3_implicit"] = 1.0  # 1.0 / (c.fe3 + 1e-3)

    K_alpha = 0.2
    limiters["alpha_explicit"] = c.so4.value / (c.so4.value + K_alpha)
    limiters["alpha_implicit"] = 1.0 / (c.so4.value + K_alpha)

    # update k-values
    k.fe3_h2s = calculate_k_iron_reduction(c.fe3.value, c.h2s.value)

    # 3. RUN PROCESSES
    # ----------------
    # Each function updates LHS, RHS, and RATES in place

    sulfate_reduction(c, k, limiters, LHS, RHS, RATES, mp)
    aerobic_respiration(c, k, limiters, LHS, RHS, RATES, mp)
    iron_reduction_h2s(c, k, limiters, LHS, RHS, RATES, mp)
    fes_oxidation(c, k, limiters, LHS, RHS, RATES, mp)
    h2s_oxidation(c, k, limiters, LHS, RHS, RATES, mp)
    pyrite_formation_s0(c, k, limiters, LHS, RHS, RATES, mp)
    pyrite_formation_h2s(c, k, limiters, LHS, RHS, RATES, mp)
    pyrite_oxidation(c, k, limiters, LHS, RHS, RATES, mp)

    # 4. FINALIZE
    # -----------
    # Pack results into f container
    for s in species_list:
        setattr(f, s, (LHS[s], RHS[s], RATES[s]))

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


# =============================================================================
# PROCESS FUNCTIONS (The Biogeochemistry)
# =============================================================================


def sulfate_reduction(c, k, lim, LHS, RHS, RATES, mp):
    """Calculate sulfate reduction.

    Reaction: 2 POC + 1 SO4 -> 1 H2S Ref: POC (k.poc_so4)

    Note that k_poc_so4 is the rate poc is being consumed, not the rate sulfate is being
    consumed. Since we consumed 2 POC for each sulfate, we devide the sulfate rate by 1/2

    Note the use of lim["so4_explicit"] in the H2S term wich contains c.so4!  so there
    is no need to multiply by c.so4

    For the isotope equations: Note that the denominator must always reference the
    explicit concentrations from the previous time step, it cannot include the implicit
    terms.  Note the addition of the 1e-20 to avoid a division by zero
    """
    phi = mp.phi
    # Scaling factor for Solid Species in Porewater-Driven Reactions
    # Assuming Rate is Intrinsic Porewater Rate (R_pw).
    # Bulk Rate = phi * R_pw. Note, this is done in the solver function
    # This is also true for solid species, but to get the correct concentrations
    # for kinetic calculations, we need to compensate for this scaling.
    # Solid Eq Term (Intrinsic) = R_pw * phi / (1-phi).
    fac_s = phi / (1.0 - phi)

    # 1. Base Rate
    rate_explicit = k.poc_so4 * c.poc * c.so4 * lim["so4_implicit"] * lim["inhib_o2"]

    # 2. POC Sink (Ref Species) - SOLID
    coeff_poc = k.poc_so4 * lim["so4_explicit"] * lim["inhib_o2"]
    add_implicit_sink(LHS, RATES, "poc", coeff_poc * fac_s, rate_explicit * fac_s)

    # 3. SO4 Sink -> Rate = 0.5 * Base - LIQUID
    coeff_so4 = k.poc_so4 * c.poc * lim["inhib_o2"] * lim["so4_implicit"]
    add_implicit_sink(LHS, RATES, "so4", coeff_so4 * 0.5, rate_explicit * 0.5)
    add_explicit_source(RHS, RATES, "h2s", rate_explicit * 0.5)

    # isotopes: fractionation disappears at low concentrations
    alpha = 1.0 + (mp.msr_alpha - 1.0) * lim["alpha_explicit"]

    # Use a larger epsilon (1e-10) and .value for substrate concentrations
    # to stabilize the sequential solver at trace levels
    s_val = c.so4.value + 1e-12
    s32_val = c.so4_32.value + 1e-12
    f_32 = alpha / (s_val + (alpha - 1) * s32_val + 1e-30)
    coeff_so4_32 = f_32 * rate_explicit

    # sulfate 32
    add_implicit_sink(
        LHS, RATES, "so4_32", coeff_so4_32 * 0.5, coeff_so4_32 * c.so4_32 * 0.5
    )
    # sulfide 32
    add_explicit_source(RHS, RATES, "h2s_32", coeff_so4_32 * c.so4_32 * 0.5)


def aerobic_respiration(c, k, lim, LHS, RHS, RATES, mp):
    """Define POC consumption by aerobic respiration.

    Reaction: 1 POC + 1.27 O2 -> 1 CO2
    Ref: POC (k.poc_o2)
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    rate_base = k.poc_o2 * c.poc * c.o2

    # POC Sink - SOLID
    coeff_poc = k.poc_o2 * c.o2
    add_implicit_sink(LHS, RATES, "poc", coeff_poc * fac_s, rate_base * fac_s)

    # O2 Sink (1.27x) - LIQUID
    coeff_o2 = 1.27 * k.poc_o2 * c.poc
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, 1.27 * rate_base)


def iron_reduction_h2s(c, k, lim, LHS, RHS, RATES, mp):
    """Define iron reduction by h2s.

    Reaction: 1 Fe3 + 1.5 H2S -> 1 FeS + 0.5 S0
    Ref: Fe3 (k.fe3_h2s)
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # Fe3 Sink - SOLID
    coeff_fe3 = k.fe3_h2s * c.h2s
    add_implicit_sink(LHS, RATES, "fe3", coeff_fe3 * fac_s, coeff_fe3 * c.fe3 * fac_s)

    # H2S Sink (1.5x) - LIQUID
    coeff_h2s = k.fe3_h2s * c.fe3
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s * 1.5, coeff_h2s * c.h2s * 1.5)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s * 1.5, coeff_h2s * c.h2s_32 * 1.5)

    # FeS Source (1.0x) - SOLID
    rate_fes = k.fe3_h2s * c.fe3 * c.h2s * lim["fe3_explicit"]
    rate_fes_32 = k.fe3_h2s * c.fe3 * c.h2s_32 * lim["fe3_explicit"]
    add_explicit_source(RHS, RATES, "fes", rate_fes * fac_s)
    add_explicit_source(RHS, RATES, "fes_32", rate_fes_32 * fac_s)

    # S0 Source (0.5x) - SOLID
    s0_rate = k.fe3_h2s * c.fe3 * c.h2s
    s0_32_rate = k.fe3_h2s * c.fe3 * c.h2s_32
    add_explicit_source(RHS, RATES, "s0", s0_rate * 0.5 * fac_s)
    add_explicit_source(RHS, RATES, "s0_32", s0_32_rate * 0.5 * fac_s)


def fes_oxidation(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 FeS + 2.25 O2 -> 1 Fe3 + 1 SO4
    Ref: FeS (k.fes_ox)
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    rate_base = k.fes_ox * c.fes * c.o2
    rate_base_32 = k.fes_ox * c.fes_32 * c.o2

    # FeS Sink - SOLID
    coeff_fes = k.fes_ox * c.o2
    add_implicit_sink(LHS, RATES, "fes", coeff_fes * fac_s, rate_base * fac_s)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes * fac_s, rate_base_32 * fac_s)

    # O2 Sink (2.25x) - LIQUID
    coeff_o2 = 2.25 * k.fes_ox * c.fes
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, rate_base * 2.25)

    # Fe3 Source (1.0x) - SOLID
    add_explicit_source(RHS, RATES, "fe3", rate_base * fac_s)

    # SO4 Source (1.0x) - LIQUID
    add_explicit_source(RHS, RATES, "so4", rate_base)
    add_explicit_source(RHS, RATES, "so4_32", rate_base_32)


def h2s_oxidation(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 H2S + 0.5 O2 -> 1 S0
    Ref: H2S (k.h2s_ox)
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # H2S Sink - LIQUID
    coeff_h2s = k.h2s_ox * c.o2
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s_32)

    # O2 Sink (0.5x) - LIQUID
    coeff_o2 = k.h2s_ox * c.h2s
    add_implicit_sink(LHS, RATES, "o2", coeff_o2 * 0.5, coeff_o2 * c.o2 * 0.5)

    # S0 Source (1.0x) - SOLID
    rate_s0 = k.h2s_ox * c.h2s * c.o2
    rate_s0_32 = k.h2s_ox * c.h2s_32 * c.o2
    add_explicit_source(RHS, RATES, "s0", rate_s0 * fac_s)
    add_explicit_source(RHS, RATES, "s0_32", rate_s0_32 * fac_s)


def pyrite_formation_s0(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 FeS + 1 S0 -> 1 FeS2
    Ref: S0 (k.s0_fes)
    """

    phi = mp.phi
    # Solid-Solid reactions do not need phi/(1-phi) scaling as they are intrinsically solid-phase.
    # Just let solver apply (1-phi) scaling.
    # Wait, if rate is calculated as k * fes * s0 (where k is solid rate cst), then
    # Bulk Rate = (1-phi) * R.
    # Solver applies (1-phi). So we pass R unscaled.
    # BUT, pyrite_formation_h2s involves H2S (liquid).

    # S0 Sink - SOLID
    coeff_s0 = k.fes_s0 * c.fes
    add_implicit_sink(LHS, RATES, "s0", coeff_s0, coeff_s0 * c.s0)
    add_implicit_sink(LHS, RATES, "s0_32", coeff_s0, coeff_s0 * c.s0_32)

    # FeS Sink (1.0x) - SOLID
    coeff_fes = k.fes_s0 * c.s0
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, coeff_fes * c.fes)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, coeff_fes * c.fes_32)

    # FeS2 Source (1.0x) - SOLID
    fes2_rate = k.fes_s0 * c.fes * c.s0
    fes2_32_rate = coeff_s0 * c.s0_32 + coeff_fes * c.fes_32
    add_explicit_source(RHS, RATES, "fes2", fes2_rate)
    add_explicit_source(RHS, RATES, "fes2_32", fes2_32_rate)


def pyrite_formation_h2s(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 FeS + 1 H2S -> 1 FeS2
    Ref: FeS (k.fes_h2s)
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # FeS Sink - SOLID
    coeff_fes = k.fes_h2s * c.h2s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes * fac_s, coeff_fes * c.fes * fac_s)
    add_implicit_sink(
        LHS, RATES, "fes_32", coeff_fes * fac_s, coeff_fes * c.fes_32 * fac_s
    )

    # H2S Sink (1.0x) - LIQUID
    coeff_h2s = k.fes_h2s * c.fes
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s_32)

    # FeS2 Source (1.0x) - SOLID
    add_explicit_source(RHS, RATES, "fes2", k.fes_h2s * c.h2s * c.fes * fac_s)
    fes2_32_rate = (coeff_fes * c.fes_32 + coeff_h2s * c.h2s_32) * fac_s
    # Wait, H2S coeff shouldn't be scaled for H2S eq, but should be for FeS2 eq?
    # Actually fes2_32 rate eqn is sum of two sources. Can we separate?
    # Re-calculate carefully for FeS2 (Solid). It needs * fac_s.

    # term 1: FeS (Solid) -> FeS2 (Solid). Rate propto FeS sink.
    term1 = coeff_fes * c.fes_32 * fac_s  # Already scaled above

    # term 2: H2S (Liquid) -> FeS2 (Solid). H2S sink (coeff_h2s * h2s_32) is liquid-unit rate.
    # We need to add this mass to Solid. So scale by fac_s.
    term2 = coeff_h2s * c.h2s_32 * fac_s

    fes2_32_rate = term1 + term2

    # Wait, coeff_fes above is scaled by fac_s. So term1 includes fac_s * fac_s?
    # No. coeff_fes in implicit sink was scaled.
    # Let's use raw vars for explicit calc clearly.

    raw_coeff_fes = k.fes_h2s * c.h2s
    raw_coeff_h2s = k.fes_h2s * c.fes

    term1_final = raw_coeff_fes * c.fes_32 * fac_s
    term2_final = raw_coeff_h2s * c.h2s_32 * fac_s

    add_explicit_source(RHS, RATES, "fes2_32", term1_final + term2_final)


def pyrite_oxidation(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 FeS2 + 3.5 O2 -> 1 Fe3 + 2 SO4
    Ref: FeS2 (k.fes2_ox)
    """

    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # FeS2 Sink - SOLID
    coeff_fes2 = k.fes2_ox * c.o2
    add_implicit_sink(
        LHS, RATES, "fes2", coeff_fes2 * fac_s, coeff_fes2 * c.fes2 * fac_s
    )
    add_implicit_sink(
        LHS, RATES, "fes2_32", coeff_fes2 * fac_s, coeff_fes2 * c.fes2_32 * fac_s
    )

    # O2 Sink (3.5x) - LIQUID
    coeff_o2 = k.fes2_ox * c.fes2
    add_implicit_sink(LHS, RATES, "o2", coeff_o2 * 3.5, coeff_o2 * c.o2 * 3.5)

    # Fe3 Source (1.0x) - SOLID
    rate_fe3 = k.fes2_ox * c.fes2 * c.o2
    add_explicit_source(RHS, RATES, "fe3", rate_fe3 * fac_s)

    # SO4 Source (2.0x) - LIQUID
    """Note that c.fes2_32 tracks the number of 32S atoms in pyrite, and not the total
    number of sulfur atoms. So unlike the FeS2, it does not contain two sulfur atoms,
    therefore we do not mutiply by 2"""
    rate_fe3_32 = k.fes2_ox * c.fes2_32 * c.o2
    add_explicit_source(RHS, RATES, "so4", rate_fe3 * 2)
    add_explicit_source(RHS, RATES, "so4_32", rate_fe3_32)
