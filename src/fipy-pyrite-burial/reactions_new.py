"""Define diagentic reactions."""


def diagenetic_reactions(mp, c, k, f):
    """Define diagenetic reactions.

    Main orchestrator for diagenetic reactions.
    Calculates limiters, initializes matrices, and calls specific process functions.
    """
    import numpy as np

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
    limiters["inhib_o2"] = eps / (c.o2 + eps)

    # Sulfate Limiter (Implicit 1/[S+K] and Explicit [S]/[S+K])
    K_so4 = 0.1
    limiters["so4_implicit"] = 1.0 / (c.so4 + K_so4)
    limiters["so4_32_implicit"] = 1.0 / (c.so4_32 + K_so4)

    limiters["so4_explicit"] = c.so4 / (c.so4 + K_so4)
    limiters["so4_32_explicit"] = c.so4_32 / (c.so4_32 + K_so4)

    limiters["fe3_explicit"] = c.fe3 / (c.fe3 + 1e-3)
    limiters["fe3_implicit"] = 1.0 / (c.fe3 + 1e-3)

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
    # 1. Base Rate
    rate_explicit = k.poc_so4 * c.poc * c.so4 * lim["so4_implicit"] * lim["inhib_o2"]

    # 2. POC Sink (Ref Species)
    coeff_poc = k.poc_so4 * lim["so4_explicit"] * lim["inhib_o2"]
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, rate_explicit)

    # 3. SO4 Sink -> Rate = 0.5 * Base
    coeff_so4 = k.poc_so4 * c.poc * lim["inhib_o2"] * lim["so4_implicit"]
    add_implicit_sink(LHS, RATES, "so4", coeff_so4 * 0.5, rate_explicit * 0.5)
    add_explicit_source(RHS, RATES, "h2s", rate_explicit * 0.5)

    # isotopes
    alpha = 1 + (mp.msr_alpha - 1) * lim["so4_explicit"]
    f_32 = alpha / (c.so4 + (alpha - 1) * c.so4_32 + 1e-30)
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
    rate_base = k.poc_o2 * c.poc * c.o2

    # POC Sink
    coeff_poc = k.poc_o2 * c.o2
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, rate_base)

    # O2 Sink (1.27x)
    coeff_o2 = 1.27 * k.poc_o2 * c.poc
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, 1.27 * rate_base)


def iron_reduction_h2s(c, k, lim, LHS, RHS, RATES, mp):
    """Define iron reduction by h2s.

    Reaction: 1 Fe3 + 1.5 H2S -> 1 FeS + 0.5 S0
    Ref: Fe3 (k.fe3_h2s)
    """
    # Fe3 Sink
    coeff_fe3 = k.fe3_h2s * c.h2s
    add_implicit_sink(LHS, RATES, "fe3", coeff_fe3, coeff_fe3 * c.fe3)

    # H2S Sink (1.5x)
    coeff_h2s = k.fe3_h2s * c.fe3
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s * 1.5, coeff_h2s * c.h2s * 1.5)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s * 1.5, coeff_h2s * c.h2s_32 * 1.5)

    # FeS Source (1.0x)
    rate_fes = k.fe3_h2s * c.fe3 * c.h2s * lim["fe3_explicit"]
    rate_fes_32 = k.fe3_h2s * c.fe3 * c.h2s_32 * lim["fe3_explicit"]
    add_explicit_source(RHS, RATES, "fes", rate_fes)
    add_explicit_source(RHS, RATES, "fes_32", rate_fes_32)

    # S0 Source (0.5x)
    s0_rate = k.fe3_h2s * c.fe3 * c.h2s
    s0_32_rate = k.fe3_h2s * c.fe3 * c.h2s_32
    add_explicit_source(RHS, RATES, "s0", s0_rate * 0.5)
    add_explicit_source(RHS, RATES, "s0_32", s0_32_rate * 0.5)


def fes_oxidation(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 FeS + 2.25 O2 -> 1 Fe3 + 1 SO4
    Ref: FeS (k.fes_ox)
    """
    rate_base = k.fes_ox * c.fes * c.o2
    rate_base_32 = k.fes_ox * c.fes_32 * c.o2

    # FeS Sink
    coeff_fes = k.fes_ox * c.o2
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, rate_base)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, rate_base_32)

    # O2 Sink (2.25x)
    coeff_o2 = 2.25 * k.fes_ox * c.fes
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, rate_base * 2.25)

    # Fe3 Source (1.0x)
    add_explicit_source(RHS, RATES, "fe3", rate_base)

    # SO4 Source (1.0x)
    add_explicit_source(RHS, RATES, "so4", rate_base)
    add_explicit_source(RHS, RATES, "so4_32", rate_base_32)


def h2s_oxidation(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 H2S + 0.5 O2 -> 1 S0
    Ref: H2S (k.h2s_ox)
    """
    # H2S Sink
    coeff_h2s = k.h2s_ox * c.o2
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s_32)

    # O2 Sink (0.5x)
    coeff_o2 = k.h2s_ox * c.h2s
    add_implicit_sink(LHS, RATES, "o2", coeff_o2 * 0.5, coeff_o2 * c.o2 * 0.5)

    # S0 Source (1.0x)
    rate_s0 = k.h2s_ox * c.h2s * c.o2
    rate_s0_32 = k.h2s_ox * c.h2s_32 * c.o2
    add_explicit_source(RHS, RATES, "s0", rate_s0)
    add_explicit_source(RHS, RATES, "s0_32", rate_s0_32)


def pyrite_formation_s0(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 FeS + 1 S0 -> 1 FeS2
    Ref: S0 (k.s0_fes)
    """

    # S0 Sink
    coeff_s0 = k.fes_s0 * c.fes
    add_implicit_sink(LHS, RATES, "s0", coeff_s0, coeff_s0 * c.s0)
    add_implicit_sink(LHS, RATES, "s0_32", coeff_s0, coeff_s0 * c.s0_32)

    # FeS Sink (1.0x)
    coeff_fes = k.fes_s0 * c.s0
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, coeff_fes * c.fes)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, coeff_fes * c.fes_32)

    # FeS2 Source (1.0x)
    fes2_rate = k.fes_s0 * c.fes * c.s0
    fes2_32_rate = coeff_s0 * c.s0_32 + coeff_fes * c.fes_32
    add_explicit_source(RHS, RATES, "fes2", fes2_rate)
    add_explicit_source(RHS, RATES, "fes2_32", fes2_32_rate)


def pyrite_formation_h2s(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 FeS + 1 H2S -> 1 FeS2
    Ref: FeS (k.fes_h2s)
    """
    # FeS Sink
    coeff_fes = k.fes_h2s * c.h2s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, coeff_fes * c.fes)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, coeff_fes * c.fes_32)

    # H2S Sink (1.0x)
    coeff_h2s = k.fes_h2s * c.fes
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s_32)

    # FeS2 Source (1.0x)
    add_explicit_source(RHS, RATES, "fes2", k.fes_h2s * c.h2s * c.fes)
    fes2_32_rate = coeff_fes * c.fes_32 + coeff_h2s * c.h2s_32
    add_explicit_source(RHS, RATES, "fes2_32", fes2_32_rate)


def pyrite_oxidation(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 FeS2 + 3.5 O2 -> 1 Fe3 + 2 SO4
    Ref: FeS2 (k.fes2_ox)
    """

    # FeS2 Sink
    coeff_fes2 = k.fes2_ox * c.o2
    add_implicit_sink(LHS, RATES, "fes2", coeff_fes2, coeff_fes2 * c.fes2)
    add_implicit_sink(LHS, RATES, "fes2_32", coeff_fes2, coeff_fes2 * c.fes2_32)

    # O2 Sink (3.5x)
    coeff_o2 = k.fes2_ox * c.fes2
    add_implicit_sink(LHS, RATES, "o2", coeff_o2 * 3.5, coeff_o2 * c.o2 * 3.5)

    # Fe3 Source (1.0x)
    rate_fe3 = k.fes2_ox * c.fes2 * c.o2
    add_explicit_source(RHS, RATES, "fe3", rate_fe3)

    # SO4 Source (2.0x)
    """Note that c.fes2_32 tracks the number of 32S atoms in pyrite, and not the total
    number of sulfur atoms. So unlike the FeS2, it does not contain two sulfur atoms,
    therefore we do not mutiply by 2"""
    rate_fe3_32 = k.fes2_ox * c.fes2_32 * c.o2
    add_explicit_source(RHS, RATES, "so4", rate_fe3 * 2)
    add_explicit_source(RHS, RATES, "so4_32", rate_fe3_32)
