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

    # limiters["fe3_explicit"] = 1.0  # c.fe3 / (c.fe3 + 1e-3)
    # limiters["fe3_implicit"] = 1.0  # 1.0 / (c.fe3 + 1e-3)

    # limiters["fes_explicit"] = c.fes / (c.fes + 1e-6)
    # limiters["fes_implicit"] = 1 / (c.fes + 1e-6)

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
    # Convert the CROSS dict into FiPy ImplicitSourceTerm objects and pack all
    # results into the f container as a tuple per species:
    #   f.<species> = (lhs_coeff, rhs_source, rates, cross_term)
    #
    # The CROSS dict contains entries of the form:
    #   CROSS[target_species] = [(source_name, coeff_with_fac), ...]
    # where coeff_with_fac already includes the porosity factor applied by
    # add_implicit_coupling_new().
    #
    # Each (source_name, coeff) entry becomes:
    #   ImplicitSourceTerm(coeff=coeff, var=c[source_name])
    # in the target species' equation.  Because var=c[source_name] refers to a
    # *different* species variable, FiPy treats this as an off-diagonal (cross-)
    # coupling term when all species equations are assembled into the coupled
    # block system in _assemble_coupled_equation() (solver_calls.py).
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

    # SO4 Source (1.0x) - LIQUID, Coupled to FeS (SOLID)
    # Use add_implicit_coupling_new with add_lhs_sink=False because the FeS sink
    # was already registered by the add_implicit_coupling_new call for fe3 above.
    # The "solid_2_liquid" ctype applies the (1-phi)/phi porosity factor automatically.
    add_implicit_coupling_new(
        "solid_2_liquid",
        CROSS,
        RATES,
        LHS,
        "so4",  # target: liquid
        "fes",  # source: solid
        k.fes_ox * c.o2,
        coeff_fes * c.fes,
        mp,
        c=c,
        add_lhs_sink=False,  # fes sink already added by the fe3 coupling above
    )

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
    # S0 Sink to Solid so no porosity scaling
    coeff_s0 = k.fes_s0 * c.fes
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
        # S0 is porewater → must include mp.fac_s to match bulk sink coefficient,
        # and use "liquid_2_solid" for correct volume conversion to fes2_32 (solid)

        # 1st S atom: from s0_32 (porewater) to fes2_32 (solid)
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

        # 2nd S atom: from fes_32 (solid) to fes2_32 (solid) — this was correct
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


def pyrite_formation_fes_ts2(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 1 H2S -> 1 FeS2"""

    # H2S Sink (1.0x) - LIQUID
    coeff_ts2 = k.fes_ts2 * c.fes * mp.hs_frac
    add_implicit_sink(LHS, RATES, "ts2", coeff_ts2, coeff_ts2 * c.ts2, c=c)

    # FeS2 Source (1.0x) - SOLID
    # Couple to FeS.
    # Rate = k * H2S * FeS.
    add_implicit_coupling_new(
        "solid_2_solid",
        CROSS,
        RATES,
        LHS,
        "fes2",
        "fes",
        k.fes_ts2 * c.ts2 * mp.hs_frac,
        k.fes_ts2 * c.ts2 * c.fes * mp.hs_frac,
        mp,
        c=c,
    )

    if hasattr(c, "fes_32"):
        # 1. Isotope sinks (on species that are consumed)
        # H2S_32 (liquid) sink
        add_implicit_sink(LHS, RATES, "ts2_32", coeff_ts2, coeff_ts2 * c.ts2_32, c=c)

        # 2. FeS2_32 Source (Solid) from FeS_32 (Solid)
        # 1st S atom: from FeS_32
        add_implicit_coupling_new(
            "solid_2_solid",
            CROSS,
            RATES,
            LHS,
            target_species="fes2_32",
            source_species="fes_32",
            coeff=k.fes_ts2 * c.ts2 * mp.hs_frac,
            rate=k.fes_ts2 * c.ts2 * c.fes_32 * mp.hs_frac,
            mp=mp,
            c=c,
            add_lhs_sink=True,  # Adds sink for fes_32
        )

        # 3. FeS2_32 Source (Solid) from TS2_32 (Liquid)
        # 2nd S atom: from H2S_32 (ts2_32)
        add_implicit_coupling_new(
            "liquid_2_solid",
            CROSS,
            RATES,
            LHS,
            target_species="fes2_32",
            source_species="ts2_32",
            coeff=coeff_ts2,
            rate=coeff_ts2 * c.ts2_32,
            mp=mp,
            c=c,
            add_lhs_sink=False,  # Already added by add_implicit_sink above
        )


def apply_rate_limiter(rate, var, fraction=0.5, eps=1e-12):
    """Limit rate so it doesn't consume more than a fraction of available var."""
    val = var.value if hasattr(var, "value") else var
    max_rate = val * fraction / 1.0  # Normalized dt=1 for steady state sweep
    return np.minimum(rate, np.maximum(max_rate, 0.0))


def pyrite_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Reaction: 1 FeS2 + 3.5 O2 → 1 Fe3 + 2 SO4

    Coupling strategy:
      - SO4 is cross-coupled to O2  (liquid_2_liquid, stoich 2/3.5)
      - Fe3 is cross-coupled to fes2 (solid_2_solid, stoich 1.0)
    """
    # ------------------------------------------------------------------
    # 1. Base coefficients
    # ------------------------------------------------------------------
    # O2 is porewater master: coeff_o2 is implicit sink on o2
    coeff_o2 = 3.5 * k.fes2_ox * c.fes2  # [L_pw basis]
    coeff_fes2 = k.fes2_ox * c.o2
    rate_o2 = coeff_o2 * c.o2  # mol/L_pw/s
    rate_fes2 = coeff_fes2 * c.fes2  # mol/L_pw/s

    # ------------------------------------------------------------------
    # 3. SO4 source — coupled to O2 (liquid_2_liquid)
    # stoich: 2 SO4 per 3.5 O2 → factor = 2/3.5
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        "liquid_2_liquid",
        CROSS,
        RATES,
        LHS,
        target_species="so4",
        source_species="o2",
        coeff=coeff_o2,
        rate=rate_o2,
        mp=mp,
        c=c,
        stoich_ratio=2.0 / 3.5,
    )

    # ------------------------------------------------------------------
    # 5. Fe3 source — coupled to fes2 (solid_2_solid)
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        "solid_2_solid",
        CROSS,
        RATES,
        LHS,
        target_species="fe3",
        source_species="fes2",
        coeff=coeff_fes2,
        rate=rate_fes2,
        mp=mp,
        c=c,
        stoich_ratio=1.0,
    )

    # ------------------------------------------------------------------
    # 6. Isotopes — so4_32 coupled to fes2_32, stoich=1.0 (mol-S basis)
    # ------------------------------------------------------------------
    if hasattr(c, "fes2_32"):
        f32_fes2 = c.fes2_32 / (c.fes2 + 1e-30)  # 32S fraction in pyrite

        # Implicit coeff for fes2_32 sink, scaled by isotope fraction
        coeff_fes2_32 = coeff_fes2 * f32_fes2  # drives so4_32 production

        add_implicit_coupling_new(
            "solid_2_liquid",
            CROSS,
            RATES,
            LHS,
            target_species="so4_32",
            source_species="fes2_32",
            coeff=coeff_fes2_32,
            rate=rate_fes2 * f32_fes2,
            mp=mp,
            c=c,
            stoich_ratio=1.0,  # fes2_32 in mol-S, not mol-FeS2
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


def fes_unified_reaction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Unified FeS Precipitation & Dissolution.

    Rate reporting responsibilities
    --------------------------------
    ts2     : add_implicit_coupling_new handles precipitation sink (in L_pw basis)
              add_explicit_source handles dissolution source (in L_pw basis)
    fe2_total: CROSS cross-coupling to ts2 handles precipitation sink (off-diagonal)
              add_explicit_source handles dissolution source (in L_bulk basis)
    fes     : add_implicit_coupling_new handles net precip (in L_solid basis)
              add_explicit_source(..., update_rates=False) adds dissolution to RHS only
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Current State
    # ------------------------------------------------------------------
    fe2_val = c.fe2_total.value
    fe2_pw_val = fe2_val * mp.fe2_pw_conc
    ts2_val = c.ts2.value
    hs_val = ts2_val * mp.hs_frac
    fes_val = c.fes.value

    # ------------------------------------------------------------------
    # 2. Saturation
    # ------------------------------------------------------------------
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_pw_val * hs_val) / omega_den

    # ------------------------------------------------------------------
    # 3. Regime Flags & Limiters
    # ------------------------------------------------------------------
    # is_precip = (omega >= 1.0).astype(float)
    # is_diss = (omega < 1.0).astype(float)
    # Smooth regime transition, continuous through omega = 1
    sharpness = 100.0  # increase to approach the hard switch; decrease if still spiking
    is_precip = 0.5 * (1.0 + np.tanh(sharpness * (omega - 1.0)))
    is_diss = 1.0 - is_precip
    fes_limiter = fes_val / (fes_val + 1e-4)

    # ------------------------------------------------------------------
    # 4. Asymptotic Coefficients
    # ------------------------------------------------------------------
    k_p_term = k.fes_isp * is_precip
    k_d_term = k.fes_isd * fes_limiter * is_diss

    s_bulk = k_p_term + (k_d_term * fes_val)  # explicit backward flux [mol/L_bulk/s]
    l_mult = s_bulk / omega_den  # implicit forward multiplier

    l_ts2 = l_mult * fe2_pw_val * mp.hs_frac  # implicit coeff for ts2  [L_pw basis]

    # Net precipitation in bulk units (positive = net precipitation)
    net_precip_bulk = l_ts2 * ts2_val - s_bulk  # mol/L_bulk/s

    # ------------------------------------------------------------------
    # 5a. ts2 → fes  (implicit precipitation via helper)
    # ------------------------------------------------------------------
    # Helper correctly reports:
    #   RATES["ts2"]  -= net_precip_bulk / phi      [L_pw basis]
    #   RATES["fes"]  += net_precip_bulk * phi/(1-phi)  [L_solid basis]
    add_implicit_coupling_new(
        "liquid_2_solid",
        CROSS,
        RATES,
        LHS,
        target_species="fes",
        source_species="ts2",
        coeff=l_ts2 / mp.phi,  # implicit sink coeff in L_pw basis
        rate=net_precip_bulk,  # net rate in L_bulk for RATES conversion
        mp=mp,
        c=c,
    )

    # ------------------------------------------------------------------
    # 5b. Explicit dissolution sources — ts2 and fe2_total
    # ------------------------------------------------------------------
    # add_explicit_source correctly adds to RATES in each species' own basis
    add_explicit_source(
        RHS, RATES, "ts2", s_bulk / mp.phi, update_rates=False
    )  # L_pw basis
    add_explicit_source(
        RHS, RATES, "fe2_total", s_bulk, update_rates=False
    )  # L_bulk basis

    # Dissolution removes FeS: add to RHS only — RATES["fes"] already set by helper above
    add_explicit_source(RHS, RATES, "fes", -s_bulk / (1.0 - mp.phi), update_rates=False)

    # ------------------------------------------------------------------
    # 5c. fe2_total cross-coupling to ts2 (off-diagonal sink)
    # ------------------------------------------------------------------
    # Replace independent self-sink with cross-coupling: fe2 consumption is
    # driven by ts2 concentration, preventing numerical drift.
    # CROSS entry: ImplicitSourceTerm(coeff=-l_ts2, var=c.ts2) → off-diagonal sink
    CROSS["fe2_total"].append(("ts2", -l_ts2))
    RATES["fe2_total"] -= getattr(net_precip_bulk, "value", net_precip_bulk)

    # ------------------------------------------------------------------
    # 6. Isotopes (32S)
    # ------------------------------------------------------------------
    if hasattr(c, "ts2_32"):
        ts2_32_val = c.ts2_32.value
        fes_32_val = c.fes_32.value

        f32_fes = fes_32_val / (fes_val + 1e-30)
        f32_ts2 = ts2_32_val / (ts2_val + 1e-30)

        l_ts2_32 = l_ts2  # no fractionation

        # Regime-dependent dissolution ratio (see isotope fix)
        # Decompose s_bulk into physical components
        s_stab = k_p_term  # numerical stabilisation → releases porewater isotope ratio
        s_diss = k_d_term * fes_val  # real dissolution → releases solid isotope ratio

        # Apply correct ratio to each component independently
        # This is correct for BOTH hard and smooth regime flags
        s_32_diss = s_stab * f32_ts2 + s_diss * f32_fes
        net_precip_32 = l_ts2_32 * ts2_32_val - s_32_diss

        # ts2_32 → fes_32 implicit coupling
        add_implicit_coupling_new(
            "liquid_2_solid",
            CROSS,
            RATES,
            LHS,
            target_species="fes_32",
            source_species="ts2_32",
            coeff=l_ts2_32 / mp.phi,
            rate=net_precip_32,
            mp=mp,
            c=c,
        )

        # Explicit dissolution
        add_explicit_source(RHS, RATES, "ts2_32", s_32_diss / mp.phi)
        add_explicit_source(
            RHS, RATES, "fes_32", -s_32_diss / (1.0 - mp.phi), update_rates=False
        )


def get_total_delta(c, mp, index=-1):
    """Get total delta that is being buried.

    Calculate the total amount of S and S32 that leaves the system through the lower
    boundary. Note that we have to count the mass of FeS2 since it has 2 S, however,
    FeS2_32 is already corrected, so we do not mutiply it.
    """

    from diff_lib import get_delta

    phi = mp.phi
    f_s = 1.0 - phi

    # Liquid species are scaled by porosity (phi)
    # Solid species are scaled by solid fraction (1-phi)
    s = phi * (c.so4.value[index] + c.ts2.value[index]) + f_s * (
        c.s0.value[index] + c.fes.value[index] + 2 * c.fes2.value[index]
    )
    s32 = phi * (c.so4_32.value[index] + c.ts2_32.value[index]) + f_s * (
        c.s0_32.value[index] + c.fes_32.value[index] + c.fes2_32.value[index]
    )

    return get_delta(s, s32, mp.VCDT)
