"""Define the reactions."""

from __future__ import annotations

import numpy as np
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
from fipy.terms import explicitSourceTerm
from fipy.variables.cellVariable import CellVariable
from fipy.terms.term import Term

from diff_lib import (
    add_explicit_source,
    add_implicit_sink,
    add_implicit_coupling,
    add_implicit_coupling_new,
)


def equilibrium_reactions(mp, c, k, f, RATES, dt):
    """Instantenous reactions that are calculated after the
    transport matrix has been solved.
    """

    for r in mp.instantenous_reactions:
        r(c, k, mp, dt, RATES)

    return f, RATES


def diagenetic_reactions(mp, c, k, f):
    """
    Main orchestrator for diagenetic reactions. That are inside
    transport matrix.
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
    from fipy import ImplicitSourceTerm

    # 1. SETUP & INITIALIZATION
    # -------------------------
    species_list = list(c.keys())

    # Accumulators (The State)
    # LHS: Diagonal (Self) Coefficients (Implicit Sinks)
    LHS = {s: np.zeros_like(c.so4.value) for s in species_list}

    # CROSS: Off-Diagonal / Coupled Terms
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
    limiters["ts2_alpha_explicit"] = c.ts2.value / (c.ts2.value + 0.05)

    # 3. RUN PROCESSES
    # ----------------
    # Each function updates LHS, RHS, and RATES in place

    for r in mp.diagenetic_reactions:
        r(c, k, limiters, LHS, RHS, RATES, CROSS, mp)

    # 4. FINALIZE
    # -----------
    # Pack results into f container
    from fipy.terms.term import Term

    for s in species_list:
        lhs_coeff = LHS.get(s, 0.0)
        cross_list = CROSS.get(s, [])

        cross_term = 0.0
        if isinstance(cross_list, list):
            for source_name, coeff in cross_list:
                # Ensure numpy arrays are wrapped in CellVariable for correct rank
                val = getattr(coeff, "value", coeff)
                if hasattr(val, "shape") and val.shape != ():
                    coeff_val = CellVariable(mesh=c[s].mesh, value=val)
                else:
                    coeff_val = val
                cross_term += ImplicitSourceTerm(coeff=coeff_val, var=c[source_name])

        setattr(f, s, (lhs_coeff, RHS[s], RATES[s], cross_term))

    return f, RATES


# =============================================================================
# PROCESS FUNCTIONS (The Biogeochemistry)
# =============================================================================
def aerobic_respiration(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define POC consumption by aerobic respiration."""
    rate_base = k.poc_o2 * c.poc * c.o2

    # POC Sink - SOLID
    coeff_poc = k.poc_o2 * c.o2 * mp.fac_s
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, rate_base * mp.fac_s, c=c)

    # O2 Sink (1.27x) - LIQUID
    coeff_o2 = 1.27 * k.poc_o2 * c.poc
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, 1.27 * rate_base, c=c)
    # No produced species here (CO2 ignored)


def sulfate_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate sulfate reduction.

    Reaction: 2 POC + 1 SO4 -> 1 TS2- Ref: POC (k.poc_so4)
    """
    # 1. Base Rate
    poc_rate = k.poc_so4 * c.poc * c.so4 * lim["so4_implicit"] * lim["inhib_o2"]
    so4_rate = poc_rate * 0.5

    # 2. POC Sink (Ref Species) - SOLID
    coeff_poc = k.poc_so4 * lim["so4_explicit"] * lim["inhib_o2"] * mp.fac_s
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, poc_rate * mp.fac_s, c=c)

    # 3. SO4 Sink -> Rate = 0.5 * Base - LIQUID
    # HS- ~ 0.5 H2S,
    coeff_so4 = k.poc_so4 * c.poc * lim["inhib_o2"] * lim["so4_implicit"] * 0.5

    # 4. Sulfate reduction
    add_implicit_coupling_new(
        "liquid_2_liquid",  # type
        CROSS,  #  Off-diagonal coupling matrix
        RATES,  #  Rate reporting dictionary
        LHS,  # Diagonal matrix (implicit sinks)
        "ts2",  # species that is produced
        "so4",  # species that is consumed
        coeff_so4,  # reaction coefficient
        so4_rate,  # coeff * concentration
        mp,
        c=c,  # model parameters
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
            "ts2_32",  # species that is produced
            "so4_32",  # source species
            coeff_so4_32,  # implicit coeff for sink
            coeff_so4_32 * c.so4_32,  # explicit rate for reporting
            mp,
            c=c,
        )


def hs_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 HS- + 0.5 O2 -> 1 S0
    Note the model tracks total reduced sulfur ts2, to get HS-
    use:  [HS-] = ts2 * mp.hs_frac
    """
    # H2S Sink - LIQUID
    # Ref: H2S
    coeff_ts2 = k.hs_ox * c.o2 * mp.hs_frac

    # O2 Sink (0.5x) - LIQUID
    coeff_o2 = 0.5 * k.hs_ox * c.ts2 * mp.hs_frac
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2, c=c)

    # S0 Source (1.0x) - SOLID, Couple to H2S
    add_implicit_coupling_new(
        "liquid_2_solid",
        CROSS,
        RATES,
        LHS,
        "s0",  # species that is produced
        "ts2",  # source species
        coeff_ts2,  # implicit coeff for sink
        coeff_ts2 * c.ts2,  # explicit rate for reporting
        mp,
        c=c,
    )

    if hasattr(c, "ts2_32"):
        """Calculate Fractionation Factors using Explicit Values (Linearization)
        Note: We use .value to get numpy arrays for the denominator to avoid
        creating complex non-linear FiPy terms that slow convergence.
         FIX 1: Use coeff_ts2 (Total Coeff) as the base
        FIX 2: Do NOT divide by c.ts2_32, this is because
        f32 = f * a  * s32 / (s + (a-1) * s32)
        and f32 = coeff32 * s
        now we substitute these terms for f32 on both sides:
        coeff_32 * s32 = coeff_s * s * a * s32/ (s + (a-1) * s32)
        -> s32 appears on both sides of teh equation, so they cancel!
        solution: remove s32 on the right hand side
        f32 =  coeff_s * s * a * s32/ (s + (a-1))
        """
        alpha = 1.0 + (mp.hs_ox_alpha - 1.0) * lim["ts2_alpha_explicit"]

        # the isotope ratio is the same for HS- and ts2, so no need
        # to add mp.hs_frac
        s_val = c.ts2 + 1e-20
        s32_val = c.ts2_32 + 1e-20
        denom = s_val + (alpha - 1.0) * s32_val

        # Scaling factor for the coefficient
        # Logic: Coeff_32 = Coeff_Tot * (S_Tot * alpha / Denom)
        # We use c.ts2 (Variable) for S_Tot to keep the Jacobian accurate
        scaling_factor = c.ts2 * alpha / denom
        coeff_ts2_32 = coeff_ts2 * scaling_factor

        # S0_32 coupled to H2S_32
        add_implicit_coupling_new(
            "liquid_2_solid",
            CROSS,
            RATES,
            LHS,
            "s0_32",  # species that is produced
            "ts2_32",  # source species
            coeff_ts2_32,  # implicit coeff for sink
            coeff_ts2_32 * c.ts2_32,  # explicit rate for reporting
            mp,
            c=c,
        )


def elemental_sulfur_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Reaction: 1 S0 + 2 O2 -> 1 SO4
    Phases: S0 (Solid), O2 (Liquid), SO4 (Liquid)
    """
    # Phase conversion: Solid Rate -> Liquid Rate
    fac_l = (1.0 - mp.phi) / mp.phi

    # S0 sink (Solid)
    # Rate = k * [O2] * [S0]
    coeff_s0 = k.s0_ox * c.o2

    # O2 Sink (2.0x) - LIQUID
    # Must include fac_l because the reaction is driven by a solid concentration
    coeff_o2 = 2.0 * k.s0_ox * c.s0 * fac_l
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2, c=c)

    # SO4 Source (1.0x) - LIQUID, Coupled to S0 (SOLID)
    add_implicit_coupling_new(
        "solid_2_liquid",
        CROSS,
        RATES,
        LHS,
        "so4",  # species that is produced
        "s0",  # source species
        coeff_s0,  # implicit coeff for sink
        coeff_s0 * c.s0,  # explicit rate for reporting
        mp,
        c=c,
    )

    if hasattr(c, "s0_32"):
        # S0_32 Source (1.0x) - LIQUID, Coupled to S0_32 (SOLID)
        add_implicit_coupling_new(
            "solid_2_liquid",
            CROSS,
            RATES,
            LHS,
            "so4_32",
            "s0_32",  # source species
            coeff_s0,  # implicit coeff for sink (same as bulk!)
            coeff_s0 * c.s0_32,  # explicit rate for reporting
            mp,
            c=c,
        )


def sulfide_mediated_iron_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Fe3 iron reduction via HS-.

    Halevy at al 2023 do not track Fe2+, explicitly, and postulate that
    this reaction proceeds as 6/9 Fe3+ + HS- -> a/9 S0 + b/9 FeS + c/9 FeS2

    whereas Velde et al use
    8 Fe3+ HS- -> 8 Fe2+ + SO4

    Here we use the half reaction
    0.5 HS- + Fe3+ -> 0.5S0 + Fe2+

    Notes:
    - the model tracks Fe2+ total, which we treat as pseudo liquid
    - [HS-] = Total S2- * mp.hs_frac
    """
    # we now use the approach by Velde et al 2016
    # k.fe3_hs = calculate_k_iron_reduction(c.fe3, c.ts2 * mp.hs_frac)

    # 1. Fe3 Sink - SOLID (Linear)
    # Rate = k * [H2S] * [Fe3]
    coeff_coupling_fe2 = k.fe3_hs * c.ts2 * mp.hs_frac

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
        c=c,
    )

    # 4. Elemental sulfur - Solid (Coupled to TS2)
    # Rate = 0.5 * k * [Fe3] * [H2S] * mp.hs_frac
    coeff_ts2 = k.fe3_hs * c.fe3 * mp.hs_frac * 0.5
    add_implicit_coupling_new(
        "liquid_2_solid",
        CROSS,
        RATES,
        LHS,
        "s0",
        "ts2",
        coeff_ts2,
        coeff_ts2 * c.ts2,
        mp,
        c=c,
    )

    if hasattr(c, "ts2_32"):
        # s_val = c.ts2 + 1e-20
        # s32_val = c.ts2_32 + 1e-20
        # s_ratio = s32_val / s_val
        # Elemental sulfur 32S - Solid
        add_implicit_coupling_new(
            "liquid_2_solid",
            CROSS,
            RATES,
            LHS,
            "s0_32",
            "ts2_32",
            coeff_ts2,  # * s_ratio,
            coeff_ts2 * c.ts2_32,  # * s_ratio,
            mp,
            c=c,
        )


def sulfide_speciation_clip(c, k, mp, dt, RATES):
    """Update reporting species (h2s, hs) based on total sulfide (ts2) and pH."""
    ts2_val = c.ts2.value
    c.h2s.value[:] = ts2_val * mp.h2s_frac
    c.hs.value[:] = ts2_val * mp.hs_frac

    if hasattr(c, "ts2_32"):
        ts2_32_val = c.ts2_32.value
        if hasattr(c, "h2s_32"):
            c.h2s_32.value[:] = ts2_32_val * mp.h2s_frac
        if hasattr(c, "hs_32"):
            c.hs_32.value[:] = ts2_32_val * mp.hs_frac


def fe2_sorption_clip(c, k, mp, dt, RATES):
    """Handle Iron Partitioning algebraically.

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
    c.fe2.setValue(c.fe2_total * mp.fe2_pw_conc)
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
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, rate_base * 0.25, c=c)

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
        c=c,
    )


def fes_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 2.25 O2 -> 1 Fe3 + 1 SO4"""

    # FeS Sink - SOLID
    coeff_fes = k.fes_ox * c.o2

    add_implicit_coupling_new(
        "solid_2_solid",
        CROSS,
        RATES,
        LHS,
        "fe3",
        "fes",
        coeff_fes,
        coeff_fes * c.fes,
        mp,
        c=c,
    )

    # O2 Sink (2.25x) - LIQUID
    # Depends on FeS (Solid).
    coeff_o2_fes = 2.25 * k.fes_ox * c.fes
    rate_base = k.fes_ox * c.fes * c.o2
    # Implicit Sink for O2: coeff = 2.25 * k * FeS.
    add_implicit_sink(LHS, RATES, "o2", coeff_o2_fes, rate_base * 2.25, c=c)

    # SO4 Source (1.0x) - LIQUID
    # Couple to FeS.
    # Target Liquid. No mp.fac_s.
    add_implicit_coupling(CROSS, RATES, "so4", "fes", k.fes_ox * c.o2, rate_base, c=c)

    if hasattr(c, "fes_32"):
        rate_base_32 = k.fes_ox * c.fes_32 * c.o2
        add_implicit_coupling_new(
            "solid_2_liquid",
            CROSS,
            RATES,
            LHS,
            "so4_32",
            "fes_32",
            k.fes_ox * c.o2,
            rate_base_32,
            mp,
            c=c,
        )


def pyrite_formation_s0(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 1 S0 -> 1 FeS2.
    This is a bit tricky as we have two different S atoms into the same
    target (FeS2)
    """
    # S0 Sink Solid
    coeff_s0 = k.fes_s0 * c.fes * mp.fac_s
    add_implicit_sink(LHS, RATES, "s0", coeff_s0, coeff_s0 * c.s0, c=c)

    # FeS to FeS2 SOLID, Rate = k * FeS * S0.
    coeff_fes = k.fes_s0 * c.s0  # porosity is corrected in  add_implicit_coupling_new!
    add_implicit_coupling_new(
        "solid_2_solid",
        CROSS,
        RATES,
        LHS,
        "fes2",
        "fes",
        coeff_fes,
        coeff_fes * c.fes,
        mp,
        c=c,
    )

    if hasattr(c, "fes2_32"):
        # 1st sulfur atom from S0_32 to FeS2_32
        add_implicit_coupling_new(
            "solid_2_solid",
            CROSS,
            RATES,
            LHS,
            "fes2_32",
            "s0_32",
            k.fes_s0 * c.fes,
            k.fes_s0 * c.fes * c.s0_32,
            mp,
            c=c,
        )
        # 2nd sulfur atom from FeS_32 to FeS2_32 coeff_fes = k.fes_s0 * c.s0
        add_implicit_coupling_new(
            "solid_2_solid",
            CROSS,
            RATES,
            LHS,
            "fes2_32",
            "fes_32",
            k.fes_s0 * c.s0,
            k.fes_s0 * c.s0 * c.fes_32,
            mp,
            c=c,
        )

        # add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, coeff_fes * c.fes_32, c=c)
        # add_implicit_sink(LHS, RATES, "s0_32", coeff_s0, coeff_s0 * c.s0_32, c=c)
        # # FeS2_32 Source
        # # Sum of S0_32 and FeS_32 ?
        # # FeS2 contains 2 sulfurs.
        # # Rate FeS2_32 is purely tracking S32 mass.
        # # S32 comes from S0_32 and FeS_32.
        # # Couple to both!
        # # Term 1: from S0_32. Rate = k * FeS * S0_32. Coeff = k * FeS * mp.fac_s.
        # add_implicit_coupling(
        #     CROSS,
        #     RATES,
        #     "fes2_32",
        #     "s0_32",
        #     k.fes_s0 * c.fes * mp.fac_s,
        #     k.fes_s0 * c.fes * c.s0_32 * mp.fac_s,
        # , c=c)
        # # Term 2: from FeS_32. Rate = k * S0 * FeS_32. Coeff = k * S0 * mp.fac_s.
        # add_implicit_coupling(
        #     CROSS,
        #     RATES,
        #     "fes2_32",
        #     "fes_32",
        #     k.fes_s0 * c.s0 * mp.fac_s,
        #     k.fes_s0 * c.s0 * c.fes_32 * mp.fac_s,
        # , c=c)


def pyrite_formation_fes_ts2(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 1 H2S -> 1 FeS2"""
    # FeS Sink - SOLID
    coeff_fes = k.fes_ts2 * c.ts2 * mp.fac_s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, coeff_fes * c.fes, c=c)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, coeff_fes * c.fes_32, c=c)

    # H2S Sink (1.0x) - LIQUID
    coeff_ts2 = k.fes_ts2 * c.fes
    add_implicit_sink(LHS, RATES, "ts2", coeff_ts2, coeff_ts2 * c.ts2, c=c)
    add_implicit_sink(LHS, RATES, "ts2_32", coeff_ts2, coeff_ts2 * c.ts2_32, c=c)

    # FeS2 Source (1.0x) - SOLID
    # Couple to FeS.
    # Rate = k * H2S * FeS.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2",
        "fes",
        k.fes_ts2 * c.ts2 * mp.fac_s,
        k.fes_ts2 * c.ts2 * c.fes * mp.fac_s,
        c=c,
    )

    # FeS2_32 Source
    # From FeS_32 and H2S_32.
    # Couple to FeS_32
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "fes_32",
        k.fes_ts2 * c.ts2 * mp.fac_s,
        k.fes_ts2 * c.ts2 * c.fes_32 * mp.fac_s,
        c=c,
    )
    # Couple to H2S_32
    # Rate = k * FeS * H2S_32.
    # Target Solid, Source Liquid. Coeff Needs mp.fac_s.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes2_32",
        "ts2_32",
        mp.fac_s * k.fes_ts2 * c.fes,
        mp.fac_s * k.fes_ts2 * c.fes * c.ts2_32,
        c=c,
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
    add_implicit_sink(LHS, RATES, "fes2", coeff_fes2, coeff_fes2 * c.fes2, c=c)
    add_implicit_sink(LHS, RATES, "fes2_32", coeff_fes2, coeff_fes2 * c.fes2_32, c=c)

    # O2 Sink (3.5x) - LIQUID
    coeff_o2 = 3.5 * k.fes2_ox * c.fes2
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2, c=c)

    # Fe3 Source (1.0x) - SOLID
    # Couple to FeS2
    add_implicit_coupling(
        CROSS,
        RATES,
        "fe3",
        "fes2",
        k.fes2_ox * c.o2 * mp.fac_s,
        k.fes2_ox * c.o2 * c.fes2 * mp.fac_s,
        c=c,
    )

    # SO4 Source (2.0x) - LIQUID
    # Couple to FeS2
    add_implicit_coupling(
        CROSS,
        RATES,
        "so4",
        "fes2",
        2 * k.fes2_ox * c.o2,
        2 * k.fes2_ox * c.o2 * c.fes2,
        c=c,
    )
    # SO4_32
    add_implicit_coupling(
        CROSS,
        RATES,
        "so4_32",
        "fes2_32",
        k.fes2_ox * c.o2,
        k.fes2_ox * c.o2 * c.fes2_32,
        c=c,
    )


def fes_precipitation_clip(c, k, mp, dt, RATES):
    """Fe2+ HS- -> FeS
    Instantaneous equilibrium 'Clip' for FeS precipitation.
    Includes Solid-Phase Buffering!
    """
    import numpy as np

    # 1. Get concentrations
    fe_pw = c.fe2_total.value * mp.fe2_pw_conc
    ts2_pw = c.ts2.value

    # 2. Define Effective Ksp (CRITICAL FIX: Include hs_frac!)
    # True Equilibrium: [Fe_pw] * [HS] = K_sp * [H+]
    # [HS] = [TS2] * hs_frac  -->  [Fe_pw] * [TS2] = (K_sp * [H+]) / hs_frac
    K_eff = (k.fes_sp * k.hplus) / mp.hs_frac

    # 3. Identify Supersaturated Cells
    iap = fe_pw * ts2_pw
    mask = iap > K_eff

    if not np.any(mask):
        return RATES

    # 4. Solve BUFFERED Quadratic
    # (Fe_pw - x * f_diss) * (TS2 - x) = K_eff
    A = mp.f_diss
    B = -(fe_pw[mask] + ts2_pw[mask] * mp.f_diss)
    C = iap[mask] - K_eff

    delta = B**2 - 4.0 * A * C

    # Negative root is the physical one (subtracting mass)
    x_precip = (-B - np.sqrt(delta)) / (2.0 * A)

    # --- REPORTING LOGIC ---
    rate_report = x_precip / dt

    RATES["ts2"][mask] -= rate_report
    RATES["fe2_total"][mask] -= rate_report * mp.phi
    RATES["fes"][mask] += rate_report * mp.fac_s

    # ---------------------------------------------------------
    # 5. Calculate Isotope Mass Transfer FIRST
    # ---------------------------------------------------------
    if hasattr(c, "ts2_32"):  # FIX: Was checking for h2s_32 but variables are ts2
        # Current Porewater Ratio
        R_pw = c.ts2_32.value[mask] / (c.ts2.value[mask] + 1e-20)

        # Mass of 32S we WANT to move
        loss_32 = x_precip * R_pw

        # Apply changes
        c.ts2_32.value[mask] -= loss_32

        # Ensure your solid isotope variable matches your initialization
        if hasattr(c, "fes_32"):
            c.fes_32.value[mask] += loss_32 * mp.fac_s

    # ---------------------------------------------------------
    # 6. Update Bulk State Variables
    # ---------------------------------------------------------
    c.ts2.value[mask] -= x_precip
    c.fe2_total.value[mask] -= x_precip * mp.phi
    c.fes.value[mask] += x_precip * mp.fac_s

    return RATES


def fes_precipitation_clip_old(c, k, mp, dt, RATES):
    """Fe2+ HS- -> FeS
    Instantaneous equilibrium 'Clip' for FeS precipitation.
    Solves (Fe - x)(S - x) = Ksp for x.
    """
    import numpy as np

    # 1. Get concentrations (Create explicit copies if you need them to stay static,
    # or just be careful with update order)
    fe = c.fe2_total.value * mp.fe2_pw_conc
    hs = c.ts2.value * mp.hs_frac  # Reference to live array

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
    RATES["ts2"][mask] -= rate_report

    # For Fe2_total (Bulk units)
    RATES["fe2_total"][mask] -= rate_report * mp.phi

    # For FeS (Solid units)
    RATES["fes"][mask] += rate_report * mp.fac_s
    # ---------------------------------------------------------
    # CRITICAL FIX: Calculate Isotope Mass Transfer FIRST
    # ---------------------------------------------------------
    # if hasattr(c, "ts2_32"):
    #     # We must use hs[mask] HERE, before we subtract x_precip from it.
    #     # This gives us the fraction of the PRE-PRECIPITATION pool.

    #     # Optional: Add fractionation factor 'alpha' (e.g., 1.035 for faster 32S precip)
    #     # alpha = 1.0

    #     # Fraction of total H2S being removed
    #     frac_precip = x_precip / (hs[mask] + 1e-30)

    #     # Calculate the mass of 32S to move
    #     loss_32 = c.ts2_32.value[mask] * frac_precip  # * alpha

    #     # Update Isotope State Variables
    #     c.ts2_32.value[mask] -= loss_32
    #     c.fes_32.value[mask] += loss_32 * mp.fac_s

    if hasattr(c, "h2s_32"):
        # Current Porewater Ratio (The 'True' Chemistry)
        # R_pw = 32S / Total_S
        R_pw = c.h2s_32.value[mask] / (c.h2s.value[mask] + 1e-20)

        # The mass of 32S we WANT to move is:
        # Total_Mass_Moved * Current_Liquid_Ratio
        # (Assuming Alpha = 1.0)
        loss_32 = x_precip * R_pw

        # Apply changes
        c.h2s_32.value[mask] -= loss_32
        c.fes_32.value[mask] += loss_32 * mp.fac_s

    # ---------------------------------------------------------
    # 5. Update Bulk State Variables (AFTER Isotope Calc)
    # ---------------------------------------------------------

    # Update TS2 (Porewater) - This modifies 'hs' via reference!
    c.ts2.value[mask] -= x_precip

    # Update Fe2_total (Bulk)
    c.fe2_total.value[mask] -= x_precip * mp.phi

    # Update FeS (Solid)
    c.fes.value[mask] += x_precip * mp.fac_s

    # Diagnostic print (useful for debugging stiffness)
    # print(f"  [Equil] Cells: {np.sum(mask)} | Max precip: {np.max(x_precip):.2e}")

    return RATES


def fes_formation_only(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """FeS precipitation using Picard Linearization for strict positivity."""

    # 1. Current State
    fe2_val = c.fe2_total.value
    fe2_pw_val = fe2_val * mp.fe2_pw_conc

    ts2_val = c.ts2.value
    hs_val = ts2_val * mp.hs_frac  # mol/L_pw

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_pw_val * hs_val) / omega_den

    # 3. Precipitation Logic (Omega > 1)
    is_precip = (omega > 1.0).astype(float)
    rate_precip_total = k.fes_isp * (omega - 1.0) * is_precip  # mol/L_bulk/s

    # ---------------------------------------------------------
    # 4. Picard Linearization (Strictly Positive Implicit Sinks)
    # l_coeff = Rate / Concentration
    # ---------------------------------------------------------

    # --- Fe2_total Equation (Bulk) ---
    l_fe2 = rate_precip_total / (fe2_val + 1e-30)

    add_implicit_sink(LHS, RATES, "fe2_total", l_fe2, rate_precip_total, c=c)
    # NO EXPLICIT SOURCE!

    # --- H2S Equation (Porewater) ---
    rate_ts2_pw = rate_precip_total / mp.phi
    l_ts2 = rate_ts2_pw / (ts2_val + 1e-30)

    add_implicit_sink(LHS, RATES, "ts2", l_ts2, rate_ts2_pw, c=c)
    # NO EXPLICIT SOURCE!

    # --- FeS (Solid) Accumulation ---
    rate_fes_solid = rate_precip_total / (1.0 - mp.phi)
    l_fes_precip = rate_fes_solid / (fe2_val + 1e-30)

    add_implicit_coupling(
        CROSS, RATES, "fes", "fe2_total", l_fes_precip, rate_fes_solid, c=c
    )

    # --- Isotopes (32S) ---
    if hasattr(c, "ts2_32"):
        ts2_32_val = c.ts2_32.value * mp.hs_frac
        ts2_val_total = ts2_val * mp.hs_frac
        f32_ts2 = ts2_32_val / (ts2_val_total + 1e-30)

        # The Picard implicit coefficient for the isotope is identical to the bulk!
        l_ts2_32 = l_ts2
        rate_32_precip = rate_ts2_pw * f32_ts2

        add_implicit_sink(LHS, RATES, "ts2_32", l_ts2_32, rate_32_precip, c=c)

        # Solid Isotope Accumulation
        rate_fes_32_solid = rate_fes_solid * f32_ts2
        l_fes_32_precip = rate_fes_32_solid / (fe2_val + 1e-30)

        add_implicit_coupling(
            CROSS, RATES, "fes_32", "fe2_total", l_fes_32_precip, rate_fes_32_solid, c=c
        )


def fes_formation_only_old(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """FeS precipitation"""
    # 1. Current State
    fe2_val = c.fe2_total.value
    # CRITICAL FIX: Convert bulk liquid moles to POREWATER concentration
    fe2_pw_val = fe2_val * mp.fe2_pw_conc  #

    ts2_val = c.ts2.value
    hs_val = ts2_val * mp.hs_frac  # mol/L_pw

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_pw_val * hs_val) / omega_den

    # 3. Precipitation Logic (Omega > 1)
    is_precip = (omega > 1.0).astype(float)
    rate_precip_total = k.fes_isp * (omega - 1.0) * is_precip  # mol/L_bulk/s

    # 4. Corrected Derivatives (Slopes)
    # Must include phase scaling and the is_precip mask!
    deriv_fe2 = (k.fes_isp * (mp.fe2_pw_conc) * hs_val) / omega_den * is_precip
    deriv_ts2 = (k.fes_isp * fe2_pw_val * mp.hs_frac) / omega_den * is_precip

    # --- Fe2_total Equation (Bulk) ---
    l_fe2 = deriv_fe2
    r_fe2 = rate_precip_total - (deriv_fe2 * fe2_val)
    add_implicit_sink(LHS, RATES, "fe2_total", l_fe2, rate_precip_total, c=c)
    add_explicit_source(RHS, RATES, "fe2_total", -r_fe2)
    RATES[
        "fe2_total"
    ] -= -r_fe2  # FIX: Remove Taylor numerical artifact from reporting!

    # --- H2S Equation (Porewater) ---
    l_ts2 = deriv_ts2 / mp.phi
    r_ts2 = (rate_precip_total / mp.phi) - (l_ts2 * ts2_val)
    add_implicit_sink(LHS, RATES, "ts2", l_ts2, rate_precip_total / mp.phi, c=c)
    add_explicit_source(RHS, RATES, "ts2", -r_ts2)
    RATES["ts2"] -= -r_ts2  # FIX: Remove artifact

    # --- FeS (Solid) Accumulation ---
    l_fes_precip = deriv_fe2 / (1.0 - mp.phi)
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes",
        "fe2_total",
        l_fes_precip,
        (rate_precip_total / (1.0 - mp.phi)),
        c=c,
    )

    # --- Isotopes (32S) ---
    if hasattr(c, "ts2_32"):
        ts2_32_val = c.ts2_32.value * mp.hs_frac
        ts2_val_total = ts2_val * mp.hs_frac
        f32_ts2 = ts2_32_val / (ts2_val_total + 1e-20)

        l_ts2_32 = l_ts2
        rate_32_precip = (rate_precip_total / mp.phi) * f32_ts2
        r_ts2_32 = rate_32_precip - (l_ts2_32 * c.ts2_32.value)

        add_implicit_sink(LHS, RATES, "ts2_32", l_ts2_32, rate_32_precip, c=c)
        add_explicit_source(RHS, RATES, "ts2_32", -r_ts2_32)
        RATES["ts2_32"] -= -r_ts2_32  # FIX: Remove artifact

        l_fes_32_precip = (deriv_fe2 / (1.0 - mp.phi)) * f32_ts2
        add_implicit_coupling(
            CROSS,
            RATES,
            "fes_32",
            "fe2_total",
            l_fes_32_precip,
            (rate_32_precip * mp.phi / (1.0 - mp.phi)),
            c=c,
        )


def fes_dissolution(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """FeS dissolution."""
    # 1. Current State
    fe2_val = c.fe2_total.value
    # CRITICAL FIX: Convert to porewater concentration
    fe2_pw_val = fe2_val * mp.fe2_pw_conc

    hs_val = c.ts2.value * mp.hs_frac
    fes_val = c.fes.value  # mol/L_solid

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_pw_val * hs_val) / omega_den

    # Note: Derivatives removed here, as implicit coefficient is based on solid 'fes'

    # ----- Dissolution Logic (Omega < 1) ------- #
    is_diss = (omega <= 1.0).astype(float)
    epsilon_fes = 1e-10
    fes_limiter = fes_val / (fes_val + epsilon_fes)

    coeff_diss = k.fes_isd * (1.0 - omega) * is_diss * fes_limiter

    # Sink for FeS (Solid)
    add_implicit_sink(LHS, RATES, "fes", coeff_diss, coeff_diss * fes_val, c=c)

    # Source for Fe2_total (Bulk)
    add_implicit_coupling(
        CROSS,
        RATES,
        "fe2_total",
        "fes",
        coeff_diss * (1.0 - mp.phi),
        coeff_diss * fes_val * (1.0 - mp.phi),
        c=c,
    )

    # Source for H2S (Porewater)
    add_implicit_coupling(
        CROSS,
        RATES,
        "ts2",
        "fes",
        coeff_diss / mp.fac_s,
        coeff_diss * fes_val / mp.fac_s,
        c=c,
    )

    # --- Isotopes (32S) ---
    if hasattr(c, "ts2_32"):
        add_implicit_sink(
            LHS, RATES, "fes_32", coeff_diss, coeff_diss * c.fes_32.value, c=c
        )

        add_implicit_coupling(
            CROSS,
            RATES,
            "ts2_32",
            "fes_32",
            coeff_diss / mp.fac_s,
            coeff_diss * c.fes_32.value / mp.fac_s,
            c=c,
        )


def fes_unified_reaction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Unified FeS Precipitation & Dissolution.
    Uses unconditionally stable asymptotic formulation.
    """
    import numpy as np

    # 1. Current State
    fe2_val = c.fe2_total.value
    fe2_pw_val = fe2_val * mp.fe2_pw_conc

    ts2_val = c.ts2.value
    hs_val = ts2_val * mp.hs_frac  # mol/L_pw
    fes_val = c.fes.value

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_pw_val * hs_val) / omega_den

    # 3. Regimes & Limiters
    is_precip = (omega >= 1.0).astype(float)
    is_diss = (omega < 1.0).astype(float)
    fes_limiter = fes_val / (fes_val + 1e-10)

    # ---------------------------------------------------------
    # 4. Asymptotic Semi-Implicit Formulation
    # ---------------------------------------------------------
    k_p_term = k.fes_isp * is_precip
    k_d_term = k.fes_isd * fes_limiter * is_diss

    # Shared Explicit Source (s_coeff) -> Represents BACKWARD FLUX (Dissolution)
    s_bulk = k_p_term + (k_d_term * fes_val)

    # Shared Implicit Multiplier -> Represents FORWARD FLUX (Precipitation)
    l_mult = (k_p_term + k_d_term * fes_val) / omega_den

    # --- H2S Equation (Porewater) ---
    l_ts2 = l_mult * fe2_pw_val * mp.hs_frac
    s_ts2 = s_bulk

    add_implicit_sink(LHS, RATES, "ts2", l_ts2 / mp.phi, 0.0, c=c)
    add_explicit_source(RHS, RATES, "ts2", s_ts2 / mp.phi)

    # --- Fe2_total Equation (Bulk) ---
    l_fe2 = l_mult * hs_val * mp.fe2_pw_conc
    s_fe2 = s_bulk

    add_implicit_sink(LHS, RATES, "fe2_total", l_fe2, 0.0, c=c)
    add_explicit_source(RHS, RATES, "fe2_total", s_fe2)

    # ---------------------------------------------------------
    # 5. Reporting and Solid Phase Coupling
    # ---------------------------------------------------------
    net_precip_bulk = (l_ts2 * ts2_val) - s_ts2

    # Clean Reporting: Revert the helper's explicit addition, then apply true net rate
    RATES["ts2"] -= s_ts2 / mp.phi
    RATES["ts2"] -= net_precip_bulk / mp.phi

    RATES["fe2_total"] -= s_fe2
    RATES["fe2_total"] -= net_precip_bulk

    # --- FeS (Solid) Equation ---
    net_fes_solid_rate = net_precip_bulk / (1.0 - mp.phi)
    add_explicit_source(RHS, RATES, "fes", net_fes_solid_rate)
    # (Helper naturally adds net_fes_solid_rate to RATES["fes"], which is correct)

    # ---------------------------------------------------------
    # 6. Isotopes (32S) - CORRECTED
    # ---------------------------------------------------------
    if hasattr(c, "ts2_32"):
        ts2_32_val = c.ts2_32.value  # DO NOT multiply by hs_frac!
        fes_32_val = c.fes_32.value

        # Solid Isotope Ratio (used for dissolution flux)
        f32_fes = fes_32_val / (fes_val + 1e-30)

        # Forward flux (Precipitation) implicit coefficient is identical to bulk
        l_ts2_32 = l_ts2

        # Backward flux (Dissolution) explicit source ALWAYS takes the solid ratio
        s_ts2_32 = s_ts2 * f32_fes

        add_implicit_sink(LHS, RATES, "ts2_32", l_ts2_32 / mp.phi, 0.0, c=c)
        add_explicit_source(RHS, RATES, "ts2_32", s_ts2_32 / mp.phi)

        # Calculate true net isotope precipitation
        net_precip_32 = (l_ts2_32 * ts2_32_val) - s_ts2_32

        # Clean Reporting
        RATES["ts2_32"] -= s_ts2_32 / mp.phi
        RATES["ts2_32"] -= net_precip_32 / mp.phi

        # FeS_32 Solid Accumulation
        net_fes_32_solid = net_precip_32 / (1.0 - mp.phi)
        add_explicit_source(RHS, RATES, "fes_32", net_fes_32_solid)


def fes_unified_reaction_safe(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Unified FeS Precipitation & Dissolution.
    Uses 'Capped Picard' formulation: Pure implicit sinks capped by exact thermodynamic limits
    to guarantee stability without phantom sources or isotope mixing.
    """
    import numpy as np

    # ---------------------------------------------------------
    # 1. State Variables & Thermodynamics
    # ---------------------------------------------------------
    fe_bulk = c.fe2_total.value
    fe_pw = fe_bulk * mp.fe2_pw_conc
    ts2_pw = c.ts2.value
    hs_pw = ts2_pw * mp.hs_frac
    fes_sol = c.fes.value

    K_eff = (k.fes_sp * k.hplus) / mp.hs_frac
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe_pw * hs_pw) / omega_den

    # Calculate EXACT thermodynamic distance to equilibrium (Buffered Quadratic)
    # x_eq is the moles/L_pw of H2S that must react to reach exactly Omega = 1
    A = mp.fe2_pw_conc
    B = -(fe_pw + hs_pw * mp.fe2_pw_conc)
    C = (fe_pw * hs_pw) - K_eff
    delta = np.maximum(B**2 - 4.0 * A * C, 0.0)
    x_eq = (-B - np.sqrt(delta)) / (2.0 * A)

    # ---------------------------------------------------------
    # 2. Calculate Raw Kinetic Rates
    # ---------------------------------------------------------
    # Precipitation (mol/L_bulk/s) -> convert to Porewater rate
    rate_p_bulk_raw = k.fes_isp * np.maximum(omega - 1.0, 0.0)
    rate_p_pw_raw = rate_p_bulk_raw / mp.phi

    # Dissolution (1/s) -> convert to Solid rate -> convert to Porewater rate
    rate_d_raw_freq = (
        k.fes_isd * np.maximum(1.0 - omega, 0.0) * (fes_sol / (fes_sol + 1e-10))
    )
    rate_d_sol_raw = rate_d_raw_freq * fes_sol
    rate_d_pw_raw = rate_d_sol_raw * ((1.0 - mp.phi) / mp.phi)

    # ---------------------------------------------------------
    # 3. Apply Thermodynamic Brakes (The Ping-Pong Fix)
    # ---------------------------------------------------------
    # Cap the rate so it cannot consume more than the equilibrium distance (x_eq)
    # over the course of a characteristic time step (~1000 seconds for 16 mins)
    tau_safe = 1000.0

    max_p_pw = np.maximum(x_eq, 0.0) / tau_safe

    # Dissolution is limited by BOTH thermodynamic distance AND available solid FeS
    max_avail_sol_pw = fes_sol * ((1.0 - mp.phi) / mp.phi)
    max_d_pw = np.minimum(np.maximum(-x_eq, 0.0), max_avail_sol_pw) / tau_safe

    # CAPPED RATES (in Porewater Units: mol/L_pw/s)
    rate_p_pw = np.minimum(rate_p_pw_raw, max_p_pw)
    rate_d_pw = np.minimum(rate_d_pw_raw, max_d_pw)

    # Convert back to native phase units for the solver
    rate_p_bulk = rate_p_pw * mp.phi
    rate_p_sol = rate_p_pw * mp.phi / (1.0 - mp.phi)

    rate_d_bulk = rate_d_pw * mp.phi
    rate_d_sol = rate_d_pw * mp.phi / (1.0 - mp.phi)

    # ---------------------------------------------------------
    # 4. Pure Picard Coupling (Zero Explicit Artifacts)
    # ---------------------------------------------------------

    # --- PRECIPITATION (Strict Implicit Sinks on Liquid, Source to Solid) ---
    l_ts2_p = rate_p_pw / (ts2_pw + 1e-30)
    l_fe_p = rate_p_bulk / (fe_bulk + 1e-30)

    add_implicit_sink(LHS, RATES, "ts2", l_ts2_p, rate_p_pw, c=c)
    add_implicit_sink(LHS, RATES, "fe2_total", l_fe_p, rate_p_bulk, c=c)
    add_explicit_source(RHS, RATES, "fes", rate_p_sol)

    # --- DISSOLUTION (Strict Implicit Sink on Solid, Source to Liquid) ---
    l_fes_d = rate_d_sol / (fes_sol + 1e-30)

    add_implicit_sink(LHS, RATES, "fes", l_fes_d, rate_d_sol, c=c)
    add_explicit_source(RHS, RATES, "ts2", rate_d_pw)
    add_explicit_source(RHS, RATES, "fe2_total", rate_d_bulk)

    # ---------------------------------------------------------
    # 5. Isotopes (32S) - Perfect Physical Splitting
    # ---------------------------------------------------------
    if hasattr(c, "ts2_32"):
        ts2_32_val = c.ts2_32.value
        fes_32_val = c.fes_32.value

        # True Isotopic Ratios
        f32_ts2 = ts2_32_val / (ts2_pw + 1e-30)
        f32_fes = fes_32_val / (fes_sol + 1e-30)

        # A. Precipitation extracts EXACTLY the liquid ratio via Implicit Sink
        l_ts2_32_p = l_ts2_p
        rate_32_p_pw = rate_p_pw * f32_ts2

        add_implicit_sink(LHS, RATES, "ts2_32", l_ts2_32_p, rate_32_p_pw, c=c)
        add_explicit_source(
            RHS, RATES, "fes_32", rate_32_p_pw * mp.phi / (1.0 - mp.phi)
        )

        # B. Dissolution releases EXACTLY the solid ratio via Explicit Source
        l_fes_32_d = l_fes_d
        rate_32_d_sol = rate_d_sol * f32_fes
        rate_32_d_pw = rate_32_d_sol * (1.0 - mp.phi) / mp.phi

        add_implicit_sink(LHS, RATES, "fes_32", l_fes_32_d, rate_32_d_sol, c=c)
        add_explicit_source(RHS, RATES, "ts2_32", rate_32_d_pw)


def fes_unified_reaction_claude(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Unified FeS Precipitation & Dissolution.
    Uses unconditionally stable asymptotic formulation.
    """
    import numpy as np

    # 1. Current State
    fe2_val = c.fe2_total.value
    fe2_pw_val = fe2_val * mp.fe2_pw_conc

    ts2_val = c.ts2.value
    hs_val = ts2_val * mp.hs_frac  # mol/L_pw
    fes_val = c.fes.value

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_pw_val * hs_val) / omega_den

    # 3. Regimes & Limiters
    is_precip = (omega >= 1.0).astype(float)
    is_diss = (omega < 1.0).astype(float)
    fes_limiter = fes_val / (fes_val + 1e-10)

    # ---------------------------------------------------------
    # 4. Asymptotic Semi-Implicit Formulation
    # ---------------------------------------------------------
    k_p_term = k.fes_isp * is_precip
    k_d_term = k.fes_isd * fes_limiter * is_diss

    # Shared Explicit Source (s_coeff) -> Represents BACKWARD FLUX (Dissolution)
    s_bulk = k_p_term + (k_d_term * fes_val)

    # Shared Implicit Multiplier -> Represents FORWARD FLUX (Precipitation)
    l_mult = (k_p_term + k_d_term * fes_val) / omega_den

    # --- H2S Equation (Porewater) ---
    l_ts2 = l_mult * fe2_pw_val * mp.hs_frac
    s_ts2 = s_bulk

    add_implicit_sink(LHS, RATES, "ts2", l_ts2 / mp.phi, 0.0, c=c)
    add_explicit_source(RHS, RATES, "ts2", s_ts2 / mp.phi)

    # --- Fe2_total Equation (Bulk) ---
    l_fe2 = l_mult * hs_val * mp.fe2_pw_conc
    s_fe2 = s_bulk

    add_implicit_sink(LHS, RATES, "fe2_total", l_fe2, 0.0, c=c)
    add_explicit_source(RHS, RATES, "fe2_total", s_fe2)

    # ---------------------------------------------------------
    # 5. Reporting and Solid Phase Coupling
    # ---------------------------------------------------------
    net_precip_bulk = (l_ts2 * ts2_val) - s_ts2

    # Clean Reporting: Revert the helper's explicit addition, then apply true net rate
    RATES["ts2"] -= s_ts2 / mp.phi
    RATES["ts2"] -= net_precip_bulk / mp.phi

    RATES["fe2_total"] -= s_fe2
    RATES["fe2_total"] -= net_precip_bulk

    # --- FeS (Solid) Equation ---
    net_fes_solid_rate = net_precip_bulk / (1.0 - mp.phi)
    add_explicit_source(RHS, RATES, "fes", net_fes_solid_rate)

    # ---------------------------------------------------------
    # 6. Isotopes (32S) - FIXED
    # ---------------------------------------------------------
    if hasattr(c, "ts2_32"):
        ts2_32_val = (
            c.ts2_32.value
        )  # Total 32S in dissolved sulfide; do NOT apply hs_frac here
        fes_32_val = c.fes_32.value

        # --- Isotope ratios ---
        f32_fes = fes_32_val / (fes_val + 1e-30)  # 32S fraction in solid FeS
        f32_ts2 = ts2_32_val / (ts2_val + 1e-30)  # 32S fraction in dissolved sulfide

        # --- Forward flux (Precipitation): implicit coefficient is identical to bulk ---
        # Rate of 32S-HS precipitation = k * fe2 * hs_32 = k * fe2 * ts2_32 * hs_frac
        # Written as: l_ts2 * ts2_32_val  (l_ts2 already contains hs_frac)
        l_ts2_32 = (
            l_ts2  # ✓ correct: hs_frac absorbed into l_ts2, ts2_32 is the variable
        )

        # --- Backward flux (Dissolution or Asymptotic Stabilisation) ---
        #
        # KEY FIX: s_bulk plays DIFFERENT physical roles in each regime:
        #
        #   DISSOLUTION regime (omega < 1):
        #     s_bulk is actual FeS dissolution flux.
        #     → 32S released in proportion to the SOLID isotope ratio (f32_fes). ✓
        #     net_32 = s_bulk*(omega*f32_ts2 - f32_fes)  [correct dissolution budget]
        #
        #   PRECIPITATION regime (omega >= 1):
        #     s_bulk is a NUMERICAL stabilisation term, NOT real dissolution.
        #     → Must use POREWATER ratio (f32_ts2) so the net is correct:
        #       net_32 = l_ts2*ts2_32 - s_bulk*f32_ts2
        #              = s_bulk*omega*f32_ts2 - s_bulk*f32_ts2
        #              = s_bulk * f32_ts2 * (omega - 1)
        #              = net_bulk * f32_ts2   ✓  (no fractionation)
        #
        #   WRONG (original code used f32_fes in both regimes):
        #       net_32 = s_bulk*omega*f32_ts2 - s_bulk*f32_fes
        #       ≠ net_bulk * f32_ts2  whenever f32_fes ≠ f32_ts2
        #
        s_ts2_32 = s_bulk * (is_precip * f32_ts2 + is_diss * f32_fes)

        add_implicit_sink(LHS, RATES, "ts2_32", l_ts2_32 / mp.phi, 0.0, c=c)
        add_explicit_source(RHS, RATES, "ts2_32", s_ts2_32 / mp.phi)

        # True net isotope precipitation rate
        net_precip_32 = (l_ts2_32 * ts2_32_val) - s_ts2_32

        # Clean Reporting
        RATES["ts2_32"] -= s_ts2_32 / mp.phi
        RATES["ts2_32"] -= net_precip_32 / mp.phi

        # FeS_32 Solid Accumulation (no fractionation: gains/loses 32S at same net rate)
        net_fes_32_solid = net_precip_32 / (1.0 - mp.phi)
        add_explicit_source(RHS, RATES, "fes_32", net_fes_32_solid)
