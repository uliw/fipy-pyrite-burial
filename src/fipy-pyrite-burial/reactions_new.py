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

    sulfate_reduction(c, k, limiters, LHS, RHS, RATES, mp)
    aerobic_respiration(c, k, limiters, LHS, RHS, RATES, mp)
    iron_reduction_h2s(c, k, limiters, LHS, RHS, RATES, mp)
    h2s_oxidation(c, k, limiters, LHS, RHS, RATES, mp)
    fe2_oxidation(c, k, limiters, LHS, RHS, RATES, mp)
    fes_oxidation(c, k, limiters, LHS, RHS, RATES, mp)
    pyrite_oxidation(c, k, limiters, LHS, RHS, RATES, mp)
    pyrite_formation_s0(c, k, limiters, LHS, RHS, RATES, mp)
    pyrite_formation_fes_h2s(c, k, limiters, LHS, RHS, RATES, mp)
    iron_sulfide_formation(c, k, limiters, LHS, RHS, RATES, mp)

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
    # Scaling factor for Solid Species in Porewater-Driven Reactions
    # Assuming Rate is Intrinsic Porewater Rate (R_pw).
    # Bulk Rate = phi * R_pw. Note, this is done in the solver function
    # This is also true for solid species, but to get the correct concentrations
    # for kinetic calculations, we need to compensate for this scaling.
    # Solid Eq Term (Intrinsic) = R_pw * phi / (1-phi).
    phi = mp.phi
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

    Because we are in a neutral medium
    Reaction: 1 Fe3 + 0.5 H2S -> 1 Fe2 + 0.5 S0
    and we consider Fe2 to be a solid (i.e., sorbed to solide surfaces)
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # Fe3 Sink - SOLID
    coeff_fe3 = k.fe3_h2s * c.h2s
    add_implicit_sink(LHS, RATES, "fe3", coeff_fe3 * fac_s, coeff_fe3 * c.fe3 * fac_s)

    # H2S Sink (0.5x) - LIQUID
    coeff_h2s = 0.5 * k.fe3_h2s * c.fe3
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s_32)

    # Fe2+ Source (1.0x) - Liquid
    rate_fe2 = k.fe3_h2s * c.fe3 * c.h2s * lim["fe3_explicit"]
    add_explicit_source(RHS, RATES, "fe2", rate_fe2)

    # S0 Source (0.5x) - SOLID
    s0_rate = 0.5 * k.fe3_h2s * c.fe3 * c.h2s
    s0_32_rate = 0.5 * k.fe3_h2s * c.fe3 * c.h2s_32
    add_explicit_source(RHS, RATES, "s0", s0_rate * fac_s)
    add_explicit_source(RHS, RATES, "s0_32", s0_32_rate * fac_s)


def fe2_oxidation(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 4 FeS2+ O2 -> 1 Fe3OOH
    Ref: FeS (k.fes_ox)
    """
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    rate_base = k.fe2_ox * c.fe2 * c.o2

    # Fe2+ Sink - Liquid
    coeff_fe2 = k.fe2_ox * c.o2
    add_implicit_sink(LHS, RATES, "fe2", coeff_fe2, rate_base)

    # O2 Sink (1/4) - LIQUID
    coeff_o2 = k.fe2_ox * c.fe2
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, rate_base / 4.0)

    # Fe3 Source (1.0x) - SOLID
    add_explicit_source(RHS, RATES, "fe3", rate_base * fac_s)


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
    alpha = 1.0 + (mp.h2s_ox_alpha - 1.0) * lim["h2s_alpha_explicit"]
    s_val = c.h2s.value + 1e-12
    s32_val = c.h2s_32.value + 1e-12
    f_32 = alpha / (s_val + (alpha - 1) * s32_val + 1e-30)
    coeff_h2s_32 = f_32 * coeff_h2s * c.h2s
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s_32, coeff_h2s_32 * c.h2s_32)

    # O2 Sink (0.5x) - LIQUID
    coeff_o2 = k.h2s_ox * c.h2s
    add_implicit_sink(LHS, RATES, "o2", coeff_o2 * 0.5, coeff_o2 * c.o2 * 0.5)

    # S0 Source (1.0x) - SOLID
    rate_s0 = coeff_h2s * c.h2s
    coeff_h2s_32 = f_32 * coeff_h2s * c.h2s
    rate_s0_32 = coeff_h2s_32 * c.h2s_32
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


def pyrite_formation_fes_h2s(c, k, lim, LHS, RHS, RATES, mp):
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


def apply_rate_limiter(rate, var, fraction=0.5, eps=1e-12):
    """Limit rate so it doesn't consume more than a fraction of available var."""
    val = var.value if hasattr(var, "value") else var
    max_rate = val * fraction / 1.0  # Normalized dt=1 for steady state sweep
    return np.minimum(rate, np.maximum(max_rate, 0.0))


def iron_sulfide_formation(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: Fe2 + H2S <-> FeS
    Method: Switch Kinetic Model with Full Linearization (Taylor Expansion)
    """
    # 1. Porosity correction
    phi = mp.phi
    fac_s = phi / (1.0 - phi)  # Converts Porewater Rate -> Solid Rate

    # 2. Saturation State
    # omega_den = [H+] * Ksp
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
    # This acts as a sink for Fe2 and H2S
    # Coeff = (k_ISP * [Partner]) / Denom
    precip_coeff_fe2 = (k_isp * c.h2s / omega_den) * is_supersat
    precip_coeff_h2s = (k_isp * c.fe2 / omega_den) * is_supersat

    # B. Explicit Source Correction ( -k_ISP )
    # Because Rate = Sink - Constant, and Trans = Source - Sink
    # We add 'Constant' to RHS to represent the '-1' in (Omega - 1)
    # Effectively: Net Sink = k*Omega - k
    precip_const_correction = k_isp * is_supersat

    # =======================================================================
    # DISSOLUTION (Omega < 1)
    # Rate = k_ISD * [FeS] * (1 - Omega)  =  k_ISD*[FeS]  -  k_ISD*[FeS]*Omega
    # =======================================================================
    k_isd = k.isd_fes
    # Conversion: The paper's k_isd is for solid volume. We need porewater rate.
    # Rate_Liq = k_isd * [FeS] / fac_s (approx, check units carefully)
    conv = 1.0 / fac_s

    # A. Explicit Source Term ( k_ISD * [FeS] )
    # This is the "Maximum Dissolution Rate" if Omega were 0.
    diss_source_max = k_isd * c.fes * conv * is_undersat

    # B. Implicit Damping Term ( - k_ISD * [FeS] * Omega )
    # This reduces the source as saturation approaches 1.
    # It acts mathematically as a Sink for Fe2 and H2S.
    # Coeff = (k_ISD * [FeS] * conv * [Partner]) / Denom
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
    # NOTE: Solid equation is governed by its own mass balance.
    # Precipitation is a Source for FeS. Dissolution is a Sink for FeS.

    # Net Liquid Rate used to drive Solid (for consistency)
    # Rate_Net = (Precip_Sink - Precip_Source) - (Diss_Source - Diss_Sink)
    # We calculate explicit values for the 'RATES' vector reporting

    term_precip = precip_coeff_fe2 * c.fe2 - precip_const_correction
    term_diss = diss_source_max - diss_coeff_fe2 * c.fe2
    net_rate_liquid = term_precip - term_diss

    # Apply to FeS (Scaled by fac_s)
    # Source: Precipitation
    # Note: We use explicit source for Precip to avoid circular FeS dependency if possible,
    # or simple linearization.
    add_explicit_source(RHS, RATES, "fes", term_precip * fac_s)

    # Sink: Dissolution
    # Standard implicit sink for FeS: Rate = k_ISD * (1-Omega) * [FeS]
    # Coeff = k_ISD * (1 - Omega)
    # This is safe to keep as standard implicit because (1-Omega) is 0 to 1.
    diss_coeff_solid = k_isd * (1.0 - omega) * is_undersat
    add_implicit_sink(LHS, RATES, "fes", diss_coeff_solid, term_diss * fac_s)

    # --- ISOTOPES (32S) ---
    # Same logic applies. Damping terms must be added to 32S sinks too.
    if hasattr(c, "h2s_32"):
        # Fe2 is the partner.
        # H2S_32 Sink = Precip (driven by Fe) + Dissolution Damping (driven by Fe)
        add_implicit_sink(LHS, RATES, "h2s_32", precip_coeff_h2s + diss_coeff_h2s, 0.0)

        # H2S_32 Source
        # Precip Correction (Ratio) + Dissolution Max (from FeS_32)
        ratio_liquid = c.h2s_32 / (c.h2s + 1e-20)
        ratio_solid = c.fes_32 / (c.fes + 1e-20)

        src_precip_32 = precip_const_correction * ratio_liquid
        src_diss_32 = diss_source_max * ratio_solid  # Comes from solid!

        add_explicit_source(RHS, RATES, "h2s_32", src_precip_32 + src_diss_32)


def iron_sulfide_formation_old(c, k, lim, LHS, RHS, RATES, mp):
    """
    Reaction: 1 Fe2 + 1 H2S -> 1 FeS
    """

    # 1. Porosity correction
    phi = mp.phi
    fac_s = phi / (1.0 - phi)

    # 2. Fe2+ Sink - Liquid
    coeff_fe2 = k.fe2_h2s * c.h2s

    add_implicit_sink(LHS, RATES, "fe2", coeff_fe2, coeff_fe2 * c.h2s)

    # 3. H2S Sink - LIQUID
    coeff_h2s = k.fe2_h2s * c.fes
    add_implicit_sink(LHS, RATES, "h2s", coeff_h2s, coeff_h2s * c.h2s)

    # Calculate fraction of 32S in H2S for isotope source
    add_implicit_sink(LHS, RATES, "h2s_32", coeff_h2s, coeff_h2s * c.h2s)

    # 4. FeS Source  SOLID
    add_explicit_source(RHS, RATES, "fes", coeff_fe2 * c.h2s)
    # fes_32
    add_explicit_source(RHS, RATES, "fes_32", coeff_fe2 * c.h2s_32)


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
