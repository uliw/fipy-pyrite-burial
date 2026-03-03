"""Define the reactions."""

from __future__ import annotations

import numpy as np
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
from fipy.terms import explicitSourceTerm
from fipy.variables.cellVariable import CellVariable
from fipy.terms.term import Term


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
    LHS = {}  # Start as empty, populated by reaction functions

    # CROSS: Off-Diagonal / Coupled Terms
    # dict of list of tuples: target -> [(source_var_name, coeff_value), ...]
    CROSS = {}  # Start as empty, populated by reaction functions

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
        lhs_term = LHS.get(s, ImplicitSourceTerm(coeff=0.0, var=c[s]))
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

        setattr(f, s, (lhs_term, RHS[s], RATES[s], cross_term))

    return f, RATES


# =============================================================================
# HELPER FUNCTIONS (Matrix Math Abstraction)
# =============================================================================
def add_implicit_sink(
    LHS: dict[str, ImplicitSourceTerm | float],
    RATES: dict[str, float],
    species: str,
    coeff: Term | float | np.ndarray | CellVariable,
    rate: float | np.ndarray | CellVariable,
    c: dict[str, CellVariable],
) -> None:
    """
    Add a linear sink ``‑coeff·var`` to the matrix entry for *species*.

    The function is tolerant to the current content of ``LHS[species]``:
    * if it is already an ``ImplicitSourceTerm`` we simply add the new
      coefficient to its ``coeff`` attribute,
    * otherwise a new ``ImplicitSourceTerm`` is created.
    """
    # ------------------------------------------------------------------
    # Turn *coeff* into a FiPy ``Term`` if it is not already one
    # ------------------------------------------------------------------
    if not isinstance(coeff, Term):
        val = getattr(coeff, "value", coeff)
        if hasattr(val, "shape") and val.shape != ():
            coeff_val = CellVariable(mesh=c[species].mesh, value=val)
        else:
            coeff_val = float(val)

        # ImplicitSourceTerm(coeff=C, var=V) results in source +C*V
        # To add a sink -S*V, we set coeff = -S.
        coeff = ImplicitSourceTerm(coeff=-coeff_val, var=c[species])

    # ------------------------------------------------------------------
    # Merge with an existing entry using Term addition
    # ------------------------------------------------------------------
    if species in LHS:
        LHS[species] = LHS[species] + coeff
    else:
        LHS[species] = coeff

    # ------------------------------------------------------------------
    # Book‑keeping – used only for diagnostics
    # ------------------------------------------------------------------
    # If rate is a FiPy term or expression, extract its value
    val = getattr(rate, "value", rate)
    RATES[species] -= val


def add_implicit_coupling(x1
    CROSS: dict[str, list[tuple[str, float | np.ndarray | CellVariable]]],
    RATES: dict[str, float],
    target_species: str,
    source_species: str,
    coeff: float | np.ndarray | CellVariable,
    rate: float | np.ndarray | CellVariable,
    c: dict[str, CellVariable],
) -> None:
    """Add a linear source ``+coeff·source`` to *target_species*."""
    CROSS.setdefault(target_species, []).append((source_species, coeff))
    RATES[target_species] += getattr(rate, "value", rate)


def add_explicit_source(RHS, RATES, species, rate):
    """Add a production term to the RHS vector."""
    RHS[species] = RHS[species] + rate
    RATES[species] += getattr(rate, "value", rate)


def add_implicit_coupling_new(
    ctype, CROSS, RATES, LHS, target_species, source_species, coeff, rate, mp, c
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
    if ctype == "liquid_2_liquid":
        fac = 1.0
    elif ctype == "liquid_2_solid":
        fac = mp.fac_s
    elif ctype == "solid_2_solid":
        fac = 1.0
    elif ctype == "solid_2_liquid":
        fac = 1.0 / mp.fac_s
    else:
        raise ValueError(f"type must be l2l, l2s, s2s, s2l, not {ctype}")

    add_implicit_coupling(
        CROSS, RATES, target_species, source_species, coeff * fac, rate * fac, c=c
    )
    add_implicit_sink(LHS, RATES, source_species, coeff, rate, c=c)


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


def fes_dissolution(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """FeS dissolution only.

    Precipitation is being handled by equilibrate_fes_precipitation
    """
    # 1. Current State
    fe2_val = c.fe2_total.value
    fe2_liq_val = fe2_val * mp.f_diss  # Bulk moles in liquid
    ts2_val = c.ts2.value
    fes_val = c.fes.value  # mol/L_solid

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_liq_val * ts2_val) / omega_den

    # Derivatives (Slopes)
    deriv_fe2 = (k.fes_isp * mp.f_diss * ts2_val) / omega_den
    deriv_ts2 = (k.fes_isp * mp.f_diss * fe2_val) / omega_den

    # ----- Dissolution Logic (Omega < 1) ------- #
    is_diss = (omega <= 1.0).astype(float)
    epsilon_fes = 1e-10
    fes_limiter = fes_val / (fes_val + epsilon_fes)
    # coeff_diss in 1/s (frequency of solid dissolution)
    coeff_diss = k.fes_isd * (1.0 - omega) * is_diss * fes_limiter

    # Sink for FeS (Solid)
    add_implicit_sink(LHS, RATES, "fes", coeff_diss, coeff_diss * fes_val, c=c)

    # Source for Fe2_total (Bulk)
    # Rate_bulk = Rate_solid * (1 - phi)
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
    # Rate_pw = Rate_solid * (1 - phi) / phi = Rate_solid / fac_s
    add_implicit_coupling(
        CROSS,
        RATES,
        "ts2",
        "fes",
        coeff_diss / mp.fac_s,
        coeff_diss * fes_val / mp.fac_s,
        c=c,
    )

    # --- 7. Isotopes (32S) ---
    if hasattr(c, "ts2_32"):
        ts2_32_val = c.ts2_32.value
        f32_ts2 = ts2_32_val / (ts2_val + 1e-20)

        # Dissolution 32S (Solid Sink)
        add_implicit_sink(
            LHS, RATES, "fes_32", coeff_diss, coeff_diss * c.fes_32.value, c=c
        )

        # Dissolution 32S (Liquid Source)
        add_implicit_coupling(
            CROSS,
            RATES,
            "ts2_32",
            "fes_32",
            coeff_diss / mp.fac_s,
            coeff_diss * c.fes_32.value / mp.fac_s,
            c=c,
        )


def fes_formation_only(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """FeS precipitation
    This requires timesteps of 1 minute or less.
    """
    # 1. Current State
    fe2_val = c.fe2_total.value
    fe2_liq_val = fe2_val * mp.f_diss  # Bulk moles in liquid
    hs_val = c.ts2.value * mp.hs_frac
    fes_val = c.fes.value  # mol/L_solid

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_liq_val * hs_val) / omega_den

    # 3. Precipitation Logic (Omega > 1)
    is_precip = (omega > 1.0).astype(float)
    rate_precip_total = k.fes_isp * (omega - 1.0) * is_precip  # mol/L_bulk/s

    # Derivatives (Slopes)
    deriv_fe2 = (k.fes_isp * mp.f_diss * hs_val) / omega_den
    deriv_ts2 = (k.fes_isp * mp.f_diss * fe2_val) / omega_den

    # --- Fe2_total Equation (Bulk) ---
    l_fe2 = deriv_fe2
    r_fe2 = rate_precip_total - (deriv_fe2 * fe2_val)
    add_implicit_sink(LHS, RATES, "fe2_total", l_fe2, rate_precip_total, c=c)
    add_explicit_source(RHS, RATES, "fe2_total", -r_fe2)

    # --- H2S Equation (Porewater) ---
    # Conversion: Rate_pw = Rate_bulk / phi
    l_ts2 = deriv_ts2 / mp.phi
    r_ts2 = (rate_precip_total / mp.phi) - (l_ts2 * hs_val)
    add_implicit_sink(LHS, RATES, "ts2", l_ts2, rate_precip_total / mp.phi, c=c)
    add_explicit_source(RHS, RATES, "ts2", -r_ts2)

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
        c=c,
    )

    if hasattr(c, "ts2_32"):
        ts2_32_val = c.ts2_32.value
        ts2_val_total = c.ts2.value
        f32_ts2 = ts2_32_val / (ts2_val_total + 1e-20)

        # Precipitation 32S
        # Use same porewater scaling as TS2 total
        l_ts2_32 = l_ts2
        rate_32_precip = (rate_precip_total / mp.phi) * f32_ts2
        r_ts2_32 = rate_32_precip - (l_ts2_32 * ts2_32_val)

        add_implicit_sink(LHS, RATES, "ts2_32", l_ts2_32, rate_32_precip, c=c)
        add_explicit_source(RHS, RATES, "ts2_32", -r_ts2_32)

        # Accumulation in FeS_32 (Solid)
        # Scale: f32 * deriv_fe2 / (1-phi)
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


def fes_formation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """FeS precipitation and dissolution.

    This requires timesteps of 1 minute or less.
    """
    # 1. Current State
    fe2_val = c.fe2_total.value
    fe2_liq_val = fe2_val * mp.f_diss  # Bulk moles in liquid
    ts2_val = c.ts2.value
    fes_val = c.fes.value  # mol/L_solid

    # 2. Saturation
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_liq_val * ts2_val) / omega_den

    # 3. Precipitation Logic (Omega > 1)
    is_precip = (omega > 1.0).astype(float)
    rate_precip_total = k.fes_isp * (omega - 1.0) * is_precip  # mol/L_bulk/s

    # Derivatives (Slopes)
    deriv_fe2 = (k.fes_isp * mp.f_diss * ts2_val) / omega_den
    deriv_ts2 = (k.fes_isp * mp.f_diss * fe2_val) / omega_den

    # --- Fe2_total Equation (Bulk) ---
    l_fe2 = deriv_fe2
    r_fe2 = rate_precip_total - (deriv_fe2 * fe2_val)
    add_implicit_sink(LHS, RATES, "fe2_total", l_fe2, rate_precip_total, c=c)
    add_explicit_source(RHS, RATES, "fe2_total", -r_fe2)

    # --- H2S Equation (Porewater) ---
    # Conversion: Rate_pw = Rate_bulk / phi
    l_ts2 = deriv_ts2 / mp.phi
    r_ts2 = (rate_precip_total / mp.phi) - (l_ts2 * ts2_val)
    add_implicit_sink(LHS, RATES, "ts2", l_ts2, rate_precip_total / mp.phi, c=c)
    add_explicit_source(RHS, RATES, "ts2", -r_ts2)

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
        c=c,
    )

    # ----- Dissolution Logic (Omega < 1) ------- #
    is_diss = (omega <= 1.0).astype(float)
    epsilon_fes = 1e-10
    fes_limiter = fes_val / (fes_val + epsilon_fes)
    # coeff_diss in 1/s (frequency of solid dissolution)
    coeff_diss = k.fes_isd * (1.0 - omega) * is_diss * fes_limiter

    # Sink for FeS (Solid)
    add_implicit_sink(LHS, RATES, "fes", coeff_diss, coeff_diss * fes_val, c=c)

    # Source for Fe2_total (Bulk)
    # Rate_bulk = Rate_solid * (1 - phi)
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
    # Rate_pw = Rate_solid * (1 - phi) / phi = Rate_solid / fac_s
    add_implicit_coupling(
        CROSS,
        RATES,
        "ts2",
        "fes",
        coeff_diss / mp.fac_s,
        coeff_diss * fes_val / mp.fac_s,
        c=c,
    )
    # At the end of fes_formation, fix rate reporting
    # _c_ts2 np.float64(1.0862995735298113e-38) float64
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
    _c_ts2 = -r_ts2[i]
    _fe2 = fe2_liq_val[i]
    _ts2 = ts2_val[i]

    # --- 7. Isotopes (32S) ---
    if hasattr(c, "ts2_32"):
        ts2_32_val = c.ts2_32.value
        f32_ts2 = ts2_32_val / (ts2_val + 1e-20)

        # Precipitation 32S
        # Use same porewater scaling as TS2 total
        l_ts2_32 = l_ts2
        rate_32_precip = (rate_precip_total / mp.phi) * f32_ts2
        r_ts2_32 = rate_32_precip - (l_ts2_32 * ts2_32_val)

        add_implicit_sink(LHS, RATES, "ts2_32", l_ts2_32, rate_32_precip, c=c)
        add_explicit_source(RHS, RATES, "ts2_32", -r_ts2_32)

        # Accumulation in FeS_32 (Solid)
        # Scale: f32 * deriv_fe2 / (1-phi)
        l_fes_32_precip = (deriv_fe2 / (1.0 - mp.phi)) * f32_ts2
        add_implicit_coupling(
            CROSS,
            RATES,
            "fes_32",
            "fe2_total",
            l_fes_32_precip,
            (rate_32_precip * mp.fac_s),
            c=c,
        )

        # Dissolution 32S (Solid Sink)
        add_implicit_sink(
            LHS, RATES, "fes_32", coeff_diss, coeff_diss * c.fes_32.value, c=c
        )

        # Dissolution 32S (Liquid Source)
        add_implicit_coupling(
            CROSS,
            RATES,
            "ts2_32",
            "fes_32",
            coeff_diss / mp.fac_s,
            coeff_diss * c.fes_32.value / mp.fac_s,
            c=c,
        )


def fes_formation_old(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Model ironsulfide precipitation and dissolution.

    Reaction: Fe2 + H2S <-> FeS
    Here we use the fraction of Fe2 that is liquid.
    c.fe2_total is the lumped expression for Fe2 liquid + sorbed
    mp.f_diss is the Fe2 fraction that is dissolved

    The precipitation reaction is
    mp.fac_s * k.fes_isp * (c.fe2_liq * c.ts2/(c.hplus * k.fes_sp) -1)
    and dissolution is
    mp.fac_s * k.fes_isd * c.fes * (1 - c.fe2_liq * c.ts2/(c.hplus * k.fes_sp))

    Note that dissolution is much slower.
    """
    # 1. Get current values (numpy arrays for coefficients)
    fe2_val = c.fe2_total.value
    fe2_liq_val = fe2_val * mp.f_diss
    ts2_val = c.ts2.value

    # 2. Calculate Saturation State (Omega)
    # Omega = [Fe2+][H2S] / ([H+] * Ksp)
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_liq_val * ts2_val) / omega_den

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

    # Maximum allowed rate (90% of liquid iron OR TS2 per time step)
    # R * dt < 0.9 * C  =>  R < 0.9 * C / dt
    # Check both reactants to avoid depleting either
    max_rate_fe2 = (0.9 * fe2_liq_val) / (dt_sim + 1e-30)
    max_rate_ts2 = (0.9 * ts2_val) / (dt_sim + 1e-30)

    max_rate_allowed = np.minimum(max_rate_fe2, max_rate_ts2)

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
    coeff_fe2_sink = common_precip * ts2_val * mp.f_diss
    add_implicit_sink(
        LHS, RATES, "fe2_total", coeff_fe2_sink, coeff_fe2_sink * fe2_val, c=c
    )

    # Source (Dissolution): +k_isd * (1-Omega) * FeS [Liquid Source = Solid Rate]
    add_implicit_coupling(
        CROSS,
        RATES,
        "fe2_total",
        "fes",
        fes_diss_coeff,
        fes_diss_coeff * c.fes.value,
        c=c,
    )

    # --- H2S (Liquid) ---
    # Sink (Precipitation): -k * Omega = -(k * fe2_liq / Omega_den) * H2S
    coeff_ts2_sink = common_precip * fe2_liq_val
    add_implicit_sink(LHS, RATES, "ts2", coeff_ts2_sink, coeff_ts2_sink * ts2_val, c=c)

    # Source (): +k
    add_explicit_source(RHS, RATES, "ts2", k_isp_eff * is_precip)

    # Source (Dissolution): +k_isd * (1-Omega) * FeS
    add_implicit_coupling(
        CROSS, RATES, "ts2", "fes", fes_diss_coeff, fes_diss_coeff * c.fes.value, c=c
    )

    # --- FeS (Solid) ---
    # Source (Precipitation): +k * Omega * fac_s = +(k * H2S * f_diss / Omega_den) * fe2_total * fac_s
    coeff_fes_source = common_precip * ts2_val * mp.f_diss * mp.fac_s
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes",
        "fe2_total",
        coeff_fes_source,
        coeff_fes_source * fe2_val,
        c=c,
    )

    # Sink (Precipitation Correction): -k * fac_s
    add_explicit_source(RHS, RATES, "fes", -k_isp_eff * is_precip * mp.fac_s)

    # Sink (Dissolution): -k_isd * (1-Omega) * FeS * fac_s
    coeff_fes_sink = fes_diss_coeff * mp.fac_s
    add_implicit_sink(
        LHS, RATES, "fes", coeff_fes_sink, coeff_fes_sink * c.fes.value, c=c
    )

    # 7. Isotopes (32S)
    if hasattr(c, "ts2_32"):
        # We assume no fractionation for FeS formation/dissolution (alpha = 1.0)
        # Precipitation Rate_32 = Rate_Base * (H2S_32 / H2S)
        ts2_inv = 1.0 / (ts2_val + 1e-20)

        # Sink for H2S_32 (Precipitation):
        # Rate = [k * Omega / H2S - k / H2S] * H2S_32
        # coeff = k * (fe2_liq / Omega_den - 1 / H2S)
        coeff_ts2_32_precip = (
            k_isp_eff * (fe2_liq_val / omega_den - ts2_inv) * is_precip
        )
        coeff_ts2_32_precip = np.maximum(coeff_ts2_32_precip, 0.0)
        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            coeff_ts2_32_precip,
            coeff_ts2_32_precip * c.ts2_32.value,
            c=c,
        )

        # Source for FeS_32 (Precipitation, coupled to H2S_32):
        add_implicit_coupling(
            CROSS,
            RATES,
            "fes_32",
            "ts2_32",
            coeff_ts2_32_precip * mp.fac_s,
            coeff_ts2_32_precip * c.ts2_32.value * mp.fac_s,
            c=c,
        )

        # --- DISSOLUTION ---
        # Sink for FeS_32:
        # Rate = k_isd * (1-Omega) * FeS_32 * fac_s
        add_implicit_sink(
            LHS, RATES, "fes_32", coeff_fes_sink, coeff_fes_sink * c.fes_32.value, c=c
        )

        # Source for H2S_32 (coupled to FeS_32):
        # Rate = k_isd * (1-Omega) * FeS_32
        add_implicit_coupling(
            CROSS,
            RATES,
            "ts2_32",
            "fes_32",
            fes_diss_coeff,
            fes_diss_coeff * c.fes_32.value,
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


def equilibrate_fes_precipitation(c, k, mp, dt, RATES):
    """Fe2+ HS- -> FeS
    Instantaneous equilibrium 'Clip' for FeS precipitation.
    Solves (Fe - x)(S - x) = Ksp for x.
    """
    import numpy as np

    # 1. Get concentrations (Create explicit copies if you need them to stay static,
    # or just be careful with update order)
    fe = c.fe2_total.value * mp.f_diss
    hs = c.ts2.value  # Reference to live array

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
    hs = c.ts2.value

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
    c.ts2.value[mask] -= x_precip

    # Update Fe2_total (Bulk)
    # We removed 'x' moles/L_porewater.
    # Convert to Bulk removal: x * phi
    c.fe2_total.value[mask] -= x_precip * mp.phi

    # Add to FeS (Solid)
    # We added 'x' moles/L_porewater.
    # Convert to Solid concentration: x * (phi / (1-phi)) -> x * fac_s
    c.fes.value[mask] += x_precip * mp.fac_s

    # 6. Optional: Isotope Tracking (32S)
    if hasattr(c, "ts2_32"):
        # The fraction of S that precipitated is x / S_total
        # Remove that same fraction from 32S
        frac_precip = x_precip / (hs[mask] + 1e-30)

        loss_32 = c.ts2_32.value[mask] * frac_precip

        c.ts2_32.value[mask] -= loss_32
        c.fes_32.value[mask] += loss_32 * mp.fac_s


#  print(f"  [Equilibration] Adjusted {np.sum(mask)} cells. Max precip: {np.max(x_precip):.2e}")


def fes_formation_gpt(
    c: dict[str, CellVariable],  # concentration dictionary (fe2_total, ts2, fes, …)
    k: dict[str, float],  # kinetic constants (fes_isp, fes_isd, fes_sp, hplus, …)
    lim: dict[str, float],  # limiting‑factor scalars (e.g. "fes_explicit")
    LHS: dict[str, ImplicitSourceTerm],
    RHS: dict[
        str, explicitSourceTerm
    ],  # you already have a similar dict for explicit terms
    RATES: dict[str, float],
    CROSS: dict[str, list[tuple[str, float | np.ndarray]]],
    mp: dict[str, float],  # model parameters (fac_s, f_diss, dt_max, …)
) -> None:
    """
    Assemble the FiPy terms for the Fe‑S precipitation / dissolution system.

    The function follows the *implicit‑fast‑chemistry* strategy:
    * precipitation (Ω>1) → implicit sink on Fe²⁺ and H₂S, implicit source on FeS
    * dissolution (Ω≤1) → implicit source on Fe²⁺ and H₂S, implicit sink on FeS
    * a tiny explicit term (`add_explicit_source`) is kept for the *slow* part of
      the precipitation rate (the ``k_isp_eff`` factor after the limiter).

    All matrix entries are added via the two helper functions above, so the
    surrounding code does not have to be changed.
    """
    # -----------------------------------------------------------------
    # 0️⃣  Grab the FiPy variables (they are stored in the ``c`` dict)
    # -----------------------------------------------------------------
    fe2: CellVariable = c["fe2_total"]
    hs: CellVariable = c["ts2"]
    fes: CellVariable = c["fes"]

    # -----------------------------------------------------------------
    # 1️⃣  Current values (as plain NumPy arrays) – needed for the *non‑linear*
    #      coefficients that go into the Jacobian.
    # -----------------------------------------------------------------
    fe2_val = fe2.value
    fe2_liq = fe2_val * mp["f_diss"]  # dissolved fraction only
    hs_val = hs.value
    fes_val = fes.value

    # -----------------------------------------------------------------
    # 2️⃣  Saturation state Ω = [Fe2][H2S] / ( [H⁺]·K_sp )
    # -----------------------------------------------------------------
    omega_den = k["hplus"] * k["fes_sp"] + 1e-30
    omega = (fe2_liq * hs_val) / omega_den

    # Boolean masks (as float) – they allow us to write *one* expression that works
    # for the whole mesh.
    is_precip = (omega > 1.0).astype(float)  # 1 where Ω>1, else 0
    is_diss = 1.0 - is_precip  # complementary mask

    # -----------------------------------------------------------------
    # 3️⃣  Precipitation (fast) – implicit part
    # -----------------------------------------------------------------
    # Raw (linearised) rate: k_isp * Ω
    # We later limit it so that we never consume >90 % of a reactant in one dt.
    raw_rate = k["fes_isp"] * (omega - 1.0) * is_precip
    raw_rate = np.maximum(raw_rate, 0.0)

    # -----------------------------------------------------------------
    # 4️⃣  Limiter – keep the reaction from draining a reactant too fast
    # -----------------------------------------------------------------
    dt_sim = mp.get("dt_max", 1.0)  # fallback for steady‑state runs
    max_rate_fe2 = (0.9 * fe2_liq) / (dt_sim + 1e-30)
    max_rate_hs = (0.9 * hs_val) / (dt_sim + 1e-30)
    max_allowed = np.minimum(max_rate_fe2, max_rate_hs)

    limiter = np.minimum(1.0, max_allowed / (raw_rate + 1e-30))
    k_isp_eff = k["fes_isp"] * limiter  # *effective* kinetic constant after limiting

    # -----------------------------------------------------------------
    # 5️⃣  Coefficients that appear in the Jacobian
    # -----------------------------------------------------------------
    # Precipitation term that multiplies the *dissolved* concentrations:
    #   -k_eff * H2S / ω_den   for Fe2
    #   -k_eff * Fe2_liq / ω_den   for H2S
    common_precip = k_isp_eff * is_precip / omega_den

    coeff_fe2_sink = common_precip * hs_val * mp["f_diss"]  # ∂R/∂Fe2
    coeff_hs_sink = common_precip * fe2_liq  # ∂R/∂H2S
    coeff_fes_src = (
        common_precip * hs_val * mp["f_diss"] * mp["fac_s"]
    )  # ∂R/∂FeS (source)

    # -----------------------------------------------------------------
    # 6️⃣  Dissolution (slow) – implicit part
    # -----------------------------------------------------------------
    # Ω_diss = min(Ω, 1)  → (1‑Ω_diss) is the driving force
    omega_diss = np.minimum(omega, 1.0)
    fes_diss_coeff = k["fes_isd"] * (1.0 - omega_diss) * is_diss * lim["fes_explicit"]
    # This coefficient multiplies the solid concentration (FeS) and appears as:
    #   +k_isd*(1‑Ω)   for Fe2   (source)
    #   +k_isd*(1‑Ω)   for H2S   (source)
    #   -k_isd*(1‑Ω)   for FeS   (sink)

    # -----------------------------------------------------------------
    # 7️⃣  Assemble the implicit terms via the helpers
    # -----------------------------------------------------------------
    # ---- Fe2 (liquid) ------------------------------------------------
    add_implicit_sink(
        LHS, RATES, "fe2_total", coeff_fe2_sink, coeff_fe2_sink * fe2_val, c=c
    )  # precipitation sink
    add_implicit_coupling(
        CROSS, RATES, "fe2_total", "fes", fes_diss_coeff, fes_diss_coeff * fes_val, c=c
    )  # dissolution source

    # ---- H2S ---------------------------------------------------------
    add_implicit_sink(
        LHS, RATES, "ts2", coeff_hs_sink, coeff_hs_sink * hs_val, c=c
    )  # precipitation sink
    # Explicit part of the *fast* precipitation (the “k_isp_eff * is_precip”
    # factor that is not linearised) – kept as an explicit source so that the
    # overall rate stays exactly k_isp*(Ω‑1) after the limiter.
    add_explicit_source(RHS, RATES, "ts2", k_isp_eff * is_precip)

    add_implicit_coupling(
        CROSS, RATES, "ts2", "fes", fes_diss_coeff, fes_diss_coeff * fes_val, c=c
    )  # dissolution source

    # ---- FeS (solid) -------------------------------------------------
    # Precipitation source (coupled to Fe2)
    add_implicit_coupling(
        CROSS, RATES, "fes", "fe2_total", coeff_fes_src, coeff_fes_src * fe2_val, c=c
    )

    # Explicit correction for the fast precipitation (the “‑k_isp_eff*is_precip*fac_s” term)
    add_explicit_source(RHS, RATES, "fes", -k_isp_eff * is_precip * mp["fac_s"])

    # Dissolution sink on the solid
    coeff_fes_sink = fes_diss_coeff * mp["fac_s"]
    add_implicit_sink(LHS, RATES, "fes", coeff_fes_sink, coeff_fes_sink * fes_val, c=c)

    # -----------------------------------------------------------------
    # 8️⃣  OPTIONAL: isotope handling (32S) – unchanged apart from the
    #            helper calls (they still work with the new LHS dict)
    # -----------------------------------------------------------------
    if "ts2_32" in c:
        ts2_32: CellVariable = c["ts2_32"]
        fes_32: CellVariable = c["fes_32"]

        ts2_inv = 1.0 / (hs_val + 1e-20)

        coeff_ts2_32_precip = k_isp_eff * (fe2_liq / omega_den - ts2_inv) * is_precip
        coeff_ts2_32_precip = np.maximum(coeff_ts2_32_precip, 0.0)

        # sink on H2S_32
        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            coeff_ts2_32_precip,
            coeff_ts2_32_precip * ts2_32.value,
            c=c,
        )

        # source on FeS_32 (coupled)
        add_implicit_coupling(
            CROSS,
            RATES,
            "fes_32",
            "ts2_32",
            coeff_ts2_32_precip * mp["fac_s"],
            coeff_ts2_32_precip * ts2_32.value * mp["fac_s"],
            c=c,
        )

        # dissolution sink on FeS_32
        add_implicit_sink(
            LHS, RATES, "fes_32", coeff_fes_sink, coeff_fes_sink * fes_32.value, c=c
        )

        # source on H2S_32 (coupled)
        add_implicit_coupling(
            CROSS,
            RATES,
            "ts2_32",
            "fes_32",
            fes_diss_coeff,
            fes_diss_coeff * fes_32.value,
            c=c,
        )


def fes_formation_implicit(
    c: dict[str, CellVariable],  # concentration dictionary (fe2_total, ts2, fes, …)
    k: dict[str, float],  # kinetic constants (fes_isp, fes_isd, fes_sp, hplus, …)
    lim: dict[str, float],  # limiting‑factor scalars (e.g. "fes_explicit")
    LHS: dict[str, ImplicitSourceTerm],  # matrix‐side terms (sink / coupling)
    RHS: dict[str, explicitSourceTerm],  # explicit source terms
    RATES: dict[str, float],  # diagnostics (accumulated raw rates)
    CROSS: dict[str, list[tuple[str, float | np.ndarray]]],  # bookkeeping for couplings
    mp: dict[str, float],  # model parameters (fac_s, f_diss, dt_max, …)
) -> None:
    """
    Build the FiPy terms for the Fe‑S system using a **fully implicit** precipitation
    reaction (Fe²⁺ + H₂S ⇌ FeS(s)).

    *Precipitation* (Ω > 1) → implicit sink on Fe²⁺ and H₂S, implicit source on FeS.
    *Dissolution* (Ω ≤ 1) → implicit source on Fe²⁺ and H₂S, implicit sink on FeS.

    The fast precipitation is **entirely inside the Jacobian**, so no artificial
    90 % limiter is needed for stability.  A small explicit source
    (`add_explicit_source`) is kept only for the *non‑linear residual* that
    guarantees the exact rate `k_isp·(Ω‑1)` after the limiter is applied.
    """
    # -----------------------------------------------------------------
    # 0️⃣  Grab the FiPy variables
    # -----------------------------------------------------------------
    fe2: CellVariable = c["fe2_total"]  # dissolved + sorbed Fe²⁺ (total pool)
    hs: CellVariable = c["ts2"]  # dissolved H₂S
    fes: CellVariable = c["fes"]  # solid FeS mass pool

    # -----------------------------------------------------------------
    # 1️⃣  Current (old‑step) values – needed for the non‑linear coefficients
    # -----------------------------------------------------------------
    fe2_val = fe2.value
    fe2_liq = fe2_val * mp["f_diss"]  # only the truly dissolved fraction
    hs_val = hs.value
    fes_val = fes.value

    # -----------------------------------------------------------------
    # 2️⃣  Saturation state Ω = [Fe²⁺][H₂S] / ([H⁺]·K_sp)
    # -----------------------------------------------------------------
    omega_den = k["hplus"] * k["fes_sp"] + 1e-30
    omega = (fe2_liq * hs_val) / omega_den

    # Masks (float arrays) – enable vectorised operations on the whole mesh
    is_precip = (omega > 1.0).astype(float)  # 1 where Ω>1, else 0
    is_diss = 1.0 - is_precip  # complementary mask

    # -----------------------------------------------------------------
    # 3️⃣  **Fully implicit** precipitation term
    # -----------------------------------------------------------------
    # The *non‑linear* residual that will be added to the RHS later:
    raw_precip_residual = k["fes_isp"] * (omega - 1.0) * is_precip
    raw_precip_residual = np.maximum(raw_precip_residual, 0.0)

    # Linearised Jacobian coefficients (∂R/∂[Fe2] , ∂R/∂[HS] , ∂R/∂[FeS])
    #   R_prec = k_isp (Ω‑1)  →  ∂R/∂[Fe2] = k_isp * H2S / ω_den
    #   (the factor f_diss appears because only the dissolved Fe²⁺ participates)
    coeff_fe2 = -k["fes_isp"] * hs_val / omega_den * mp["f_diss"] * is_precip
    coeff_hs = -k["fes_isp"] * fe2_liq / omega_den * is_precip
    coeff_fes = (
        k["fes_isp"] * hs_val * mp["f_diss"] * mp["fac_s"] / omega_den * is_precip
    )

    # -------------------------------------------------------------
    # 4️⃣  Add the implicit precipitation pieces via the helpers
    # -------------------------------------------------------------
    # Fe²⁺ sink
    add_implicit_sink(LHS, RATES, "fe2_total", coeff_fe2, coeff_fe2 * fe2_val, c=c)

    # H₂S sink
    add_implicit_sink(LHS, RATES, "ts2", coeff_hs, coeff_hs * hs_val, c=c)

    # FeS source (coupled to Fe²⁺)
    add_implicit_coupling(
        CROSS, RATES, "fes", "fe2_total", coeff_fes, coeff_fes * fe2_val, c=c
    )

    # -------------------------------------------------------------
    # 5️⃣  Explicit residual for the precipitation (non‑linear part)
    # -------------------------------------------------------------
    # This term forces the *exact* rate k_isp·(Ω‑1) after the linear solve.
    add_explicit_source(RHS, RATES, "fe2_total", raw_precip_residual)
    add_explicit_source(RHS, RATES, "ts2", raw_precip_residual)
    add_explicit_source(RHS, RATES, "fes", -raw_precip_residual * mp["fac_s"])

    # -----------------------------------------------------------------
    # 6️⃣  Dissolution (slow) – still treated implicitly
    # -----------------------------------------------------------------
    # Driving force = 1‑Ω_diss  (Ω_diss = min(Ω,1))
    omega_diss = np.minimum(omega, 1.0)
    fes_diss_coeff = (
        k["fes_isd"] * (1.0 - omega_diss) * is_diss * lim["fes_explicit"]
    )  # multiplies the solid concentration

    # Fe²⁺ source from dissolution
    add_implicit_coupling(
        CROSS, RATES, "fe2_total", "fes", fes_diss_coeff, fes_diss_coeff * fes_val, c=c
    )

    # H₂S source from dissolution
    add_implicit_coupling(
        CROSS, RATES, "ts2", "fes", fes_diss_coeff, fes_diss_coeff * fes_val, c=c
    )

    # FeS sink from dissolution (scaled by the solid‑phase factor)
    coeff_fes_sink = fes_diss_coeff * mp["fac_s"]
    add_implicit_sink(LHS, RATES, "fes", coeff_fes_sink, coeff_fes_sink * fes_val, c=c)

    # -----------------------------------------------------------------
    # 7️⃣  OPTIONAL: 32S isotopes – unchanged apart from the helper calls
    # -----------------------------------------------------------------
    if "ts2_32" in c:
        ts2_32: CellVariable = c["ts2_32"]
        fes_32: CellVariable = c["fes_32"]

        # Inverse H₂S concentration (used in the isotopic formulation)
        ts2_inv = 1.0 / (hs_val + 1e-20)

        # Isotopic precipitation coefficient (same mask as the main reaction)
        coeff_ts2_32_precip = k["fes_isp"] * (fe2_liq / omega_den - ts2_inv) * is_precip
        coeff_ts2_32_precip = np.maximum(coeff_ts2_32_precip, 0.0)

        # Sink on H₂S_32 (implicit)
        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            coeff_ts2_32_precip,
            coeff_ts2_32_precip * ts2_32.value,
            c=c,
        )

        # Source on FeS_32 (coupled to H₂S_32)
        add_implicit_coupling(
            CROSS,
            RATES,
            "fes_32",
            "ts2_32",
            coeff_ts2_32_precip * mp["fac_s"],
            coeff_ts2_32_precip * ts2_32.value * mp["fac_s"],
            c=c,
        )

        # Dissolution sink on FeS_32 (same coefficient as the bulk solid)
        add_implicit_sink(
            LHS, RATES, "fes_32", coeff_fes_sink, coeff_fes_sink * fes_32.value, c=c
        )

        # Source on H₂S_32 (coupled to FeS_32 dissolution)
        add_implicit_coupling(
            CROSS,
            RATES,
            "ts2_32",
            "fes_32",
            fes_diss_coeff,
            fes_diss_coeff * fes_32.value,
            c=c,
        )


def fes_formation_fully_implicit(
    c: dict[str, CellVariable],
    k: dict[str, float],
    lim: dict[str, float],
    LHS: dict[str, ImplicitSourceTerm],
    RHS: dict[str, explicitSourceTerm],
    RATES: dict[str, float],
    CROSS: dict[str, list[tuple[str, float]]],
    mp: dict[str, float],
) -> None:
    """
    Fast FeS precipitation is treated *entirely* implicitly.
    The explicit residual is omitted because it is the source of the
    ``9e+36`` blow‑up.
    """
    fe2 = c["fe2_total"]
    hs = c["ts2"]
    fes = c["fes"]

    # ----- 0. old‑step values (used only for the Jacobian) -----
    fe2_old = fe2.value * mp["f_diss"]
    hs_old = hs.value
    fes_old = fes.value

    # ----- 1. saturation index Ω -----
    omega_den = k["hplus"] * k["fes_sp"] + 1e-30
    omega = (fe2_old * hs_old) / omega_den

    precip_mask = (omega > 1.0).astype(float)  # 1 where Ω>1, else 0
    diss_mask = 1.0 - precip_mask

    # ----- 2. Implicit precipitation coefficients (Jacobian) -----
    #   R = k_isp * (Ω‑1)   →   ∂R/∂Fe2 = k_isp * hs / ω_den
    coeff_fe2 = -k["fes_isp"] * hs_old / omega_den * mp["f_diss"] * precip_mask
    coeff_hs = -k["fes_isp"] * fe2_old / omega_den * precip_mask
    coeff_fes = k["fes_isp"] * mp["fac_s"] * precip_mask  # source on solid

    # ----- 3. Add the implicit terms via the helper utilities -----
    add_implicit_sink(
        LHS, RATES, "fe2_total", coeff_fe2, coeff_fe2 * fe2.value, c=c
    )  # sink on Fe2
    add_implicit_sink(
        LHS, RATES, "ts2", coeff_hs, coeff_hs * hs.value, c=c
    )  # sink on H2S
    add_implicit_coupling(
        CROSS, RATES, "fes", "fe2_total", coeff_fes, coeff_fes * fe2.value, c=c
    )  # solid source

    # ----- 4. Slow dissolution (still implicit) -----
    omega_diss = np.minimum(omega, 1.0)
    fes_diss_coeff = k["fes_isd"] * (1.0 - omega_diss) * diss_mask * lim["fes_explicit"]

    add_implicit_coupling(
        CROSS,
        RATES,
        "fe2_total",
        "fes",
        fes_diss_coeff,
        fes_diss_coeff * fes.value,
        c=c,
    )
    add_implicit_coupling(
        CROSS, RATES, "ts2", "fes", fes_diss_coeff, fes_diss_coeff * fes.value, c=c
    )
    coeff_fes_sink = fes_diss_coeff * mp["fac_s"]
    add_implicit_sink(
        LHS, RATES, "fes", coeff_fes_sink, coeff_fes_sink * fes.value, c=c
    )

    # ----- 5️⃣  (Optional) isotopic terms – unchanged, but make sure they are on the same mesh -----
    if "ts2_32" in c:
        ts2_32 = c["ts2_32"]
        fes_32 = c["fes_32"]
        ts2_inv = 1.0 / (hs_old + 1e-20)

        coeff_ts2_32_precip = (
            k["fes_isp"] * (fe2_old / omega_den - ts2_inv) * precip_mask
        )
        coeff_ts2_32_precip = np.maximum(coeff_ts2_32_precip, 0.0)

        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            coeff_ts2_32_precip,
            coeff_ts2_32_precip * ts2_32.value,
            c=c,
        )
        add_implicit_coupling(
            CROSS,
            RATES,
            "fes_32",
            "ts2_32",
            coeff_ts2_32_precip * mp["fac_s"],
            coeff_ts2_32_precip * ts2_32.value * mp["fac_s"],
            c=c,
        )
        add_implicit_sink(
            LHS, RATES, "fes_32", coeff_fes_sink, coeff_fes_sink * fes_32.value, c=c
        )
        add_implicit_coupling(
            CROSS,
            RATES,
            "ts2_32",
            "fes_32",
            fes_diss_coeff,
            fes_diss_coeff * fes_32.value,
            c=c,
        )


def fes_formation_fully_implicit_2(
    c: dict[str, CellVariable],
    k: dict[str, float],
    lim: dict[str, float],
    LHS: dict[str, ImplicitSourceTerm],
    RHS: dict[str, explicitSourceTerm],
    RATES: dict[str, float],
    CROSS: dict[str, list[tuple[str, float]]],
    mp: dict[str, float],
) -> None:
    """
    Assemble the FiPy matrices for the Fe‑S system with **all fast
    precipitation terms treated implicitly**.

    The implementation follows the same public API that the rest of the
    code expects:

    * ``add_implicit_sink``   – adds a linear sink (‑coeff·var) to a species.
    * ``add_implicit_coupling`` – adds a linear source that couples one
      variable to another.

    The *only* change is that the precipitation reaction
    ``Fe²⁺ + H₂S ⇌ FeS`` is written as a *single* ``ImplicitSourceTerm``
    that contains the full non‑linear expression
    ``R = k_isp·(Ω‑1)`` where

        Ω = (Fe²⁺·f_diss·H₂S) / (H⁺·K_sp)

    Because the term is fully implicit the Newton solver sees the
    correct Jacobian; no artificial “90 % limiter” or old‑step linearisation
    is required, and the large one‑year time step (dt≈3.15e7 s) remains stable.

    Parameters
    ----------
    c, k, lim, mp
        Dictionaries that hold the FiPy variables, kinetic constants,
        user‑defined limiting factors and model parameters (porosity,
        dissolution fraction, etc.).  The keys used here are exactly the
        same as in the original routine.
    LHS, RHS, RATES, CROSS
        Containers that collect the implicit/explicit source terms,
        diagnostic rate totals and the coupling information required
        later when the global FiPy equation is built.
    """
    # -----------------------------------------------------------------
    # 0️⃣  Grab the FiPy cell‑variables
    # -----------------------------------------------------------------
    fe2: CellVariable = c["fe2_total"]
    hs: CellVariable = c["ts2"]
    fes: CellVariable = c["fes"]

    # -----------------------------------------------------------------
    # 1️⃣  Build the *non‑linear* precipitation rate
    # -----------------------------------------------------------------
    # Saturation index Ω (vector‑valued, depends on the unknowns)
    omega_den = k["hplus"] * k["fes_sp"] + 1e-30
    omega = (fe2 * mp["f_diss"] * hs) / omega_den  # FiPy expression

    # Full (backward‑Euler) rate:  R = k_isp·(Ω‑1)
    rate_precip = k["fes_isp"] * (omega - 1.0)  # FiPy CellVariable

    # -----------------------------------------------------------------
    # 2️⃣  Implicit source terms for the three species
    # -----------------------------------------------------------------
    #   Fe²⁺  :  -R·f_diss   (sink)
    #   H₂S   :  -R          (sink)
    #   FeS   :  +R·fac_s    (source)
    #
    #  We create three independent ImplicitSourceTerm objects and store
    #  them in the ``LHS`` dictionary.  If a term already exists for the
    #  species we simply add the new coefficient to the existing one
    #  (FiPy allows ``term.coeff += …``).
    #
    #  NOTE:  The helper functions are *not* used for these three lines
    #         because they expect a *linear* coefficient.  The rest of
    #         the routine (slow dissolution, isotopes) continues to use
    #         the original helpers unchanged.

    # ---- Fe²⁺ -------------------------------------------------------
    #   Sink = -(-rate_precip * f_diss) * fe2 = +rate_precip * f_diss * fe2
    #   Wait, R = k_isp * (omega - 1). Sink should be -R*f_diss.
    #   ImplicitSourceTerm(coeff=C, var=V) adds C*V to the equation.
    #   Equation: Transient + ... = ... + ImplicitSourceTerm(coeff=C, var=V)
    #   So we want C = -rate_precip * f_diss / fe2 ? No.
    #   Actually rate_precip already contains fe2 (beta*fe2*hs).
    #   So we can write R = (k_isp * (omega-1)/fe2) * fe2.
    #   But omega = (fe2 * f_diss * hs) / omega_den.
    #   So R = (k_isp * f_diss * hs / omega_den) * fe2 - k_isp.
    #   This is what add_implicit_sink expects: a coefficient and a rate for diagnostics.

    coeff_precip_fe2 = k["fes_isp"] * hs * mp["f_diss"] / omega_den
    add_implicit_sink(
        LHS,
        RATES,
        "fe2_total",
        coeff=coeff_precip_fe2,
        rate=rate_precip * mp["f_diss"],
        c=c,
    )

    # ---- H₂S --------------------------------------------------------
    coeff_precip_hs = k["fes_isp"] * fe2 * mp["f_diss"] / omega_den
    add_implicit_sink(LHS, RATES, "ts2", coeff=coeff_precip_hs, rate=rate_precip, c=c)

    # ---- FeS (solid) -------------------------------------------------
    # Source on solid = +rate_precip * fac_s
    # We can treat it as a coupling or an explicit source if it doesn't depend on 'fes'
    # Since it depends on fe2 and hs, it is a coupling.
    add_implicit_coupling(
        CROSS,
        RATES,
        "fes",
        "fe2_total",
        coeff=coeff_precip_fe2 * mp["fac_s"],
        rate=rate_precip * mp["fac_s"],
        c=c,
    )
    # And we need to subtract the constant part k_isp * fac_s
    add_explicit_source(RHS, RATES, "fes", -k["fes_isp"] * mp["fac_s"])

    # -----------------------------------------------------------------
    # 3️⃣  Book‑keeping of the (non‑linear) precipitation rate
    # -----------------------------------------------------------------
    # ``RATES`` is only for diagnostics – we store the *cell‑averaged* value.
    avg_rate = rate_precip.value
    RATES["fe2_total"] += avg_rate
    RATES["ts2"] += avg_rate
    RATES["fes"] += avg_rate

    # -----------------------------------------------------------------
    # 4️⃣  Slow dissolution (still linear, therefore we keep the helpers)
    # -----------------------------------------------------------------
    #   Ω_diss = min(Ω, 1)   →  (1‑Ω_diss) is the driving force
    omega_diss = np.minimum(omega, 1.0)  # FiPy expression
    diss_mask = 1.0 - np.where(omega > 1.0, 1.0, 0.0)  # same as 1‑precip_mask

    fes_diss_coeff = (
        k["fes_isd"] * (1.0 - omega_diss) * diss_mask * lim["fes_explicit"]
    )  # this is a *linear* coefficient that multiplies the solid

    # Fe²⁺ source from dissolution
    add_implicit_coupling(
        CROSS,
        RATES,
        target_species="fe2_total",
        source_species="fes",
        coeff=fes_diss_coeff,
        rate=fes_diss_coeff * fes.value,
        c=c,
    )

    # H₂S source from dissolution
    add_implicit_coupling(
        CROSS,
        RATES,
        target_species="ts2",
        source_species="fes",
        coeff=fes_diss_coeff,
        rate=fes_diss_coeff * fes.value,
        c=c,
    )

    # FeS sink from dissolution (multiply by the solid‑phase factor)
    coeff_fes_sink = fes_diss_coeff * mp["fac_s"]
    add_implicit_sink(
        LHS, RATES, "fes", coeff_fes_sink, coeff_fes_sink * fes.value, c=c
    )

    # -----------------------------------------------------------------
    # 5️⃣  Optional isotope bookkeeping (unchanged – still uses helpers)
    # -----------------------------------------------------------------
    if "ts2_32" in c:
        ts2_32: CellVariable = c["ts2_32"]
        fes_32: CellVariable = c["fes_32"]

        # 5a – precipitation (non‑linear part) for 32S
        ts2_inv = 1.0 / (hs.value + 1e-20)

        coeff_ts2_32_precip = (
            k["fes_isp"]
            * (fe2 * mp["f_diss"] / omega_den - ts2_inv)
            * np.where(omega > 1.0, 1.0, 0.0)
        )
        coeff_ts2_32_precip = np.maximum(coeff_ts2_32_precip, 0.0)

        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            coeff_ts2_32_precip,
            coeff_ts2_32_precip * ts2_32.value,
            c=c,
        )

        # 5b – coupling of 32S to the solid
        add_implicit_coupling(
            CROSS,
            RATES,
            target_species="fes_32",
            source_species="ts2_32",
            coeff=coeff_ts2_32_precip * mp["fac_s"],
            rate=coeff_ts2_32_precip * ts2_32.value * mp["fac_s"],
            c=c,
        )

        # 5c – dissolution sink on the solid (same coeff as for the main solid)
        add_implicit_sink(
            LHS, RATES, "fes_32", coeff_fes_sink, coeff_fes_sink * fes_32.value, c=c
        )

        # 5d – dissolution source on H₂S_32 (coupled to solid)
        add_implicit_coupling(
            CROSS,
            RATES,
            target_species="ts2_32",
            source_species="fes_32",
            coeff=fes_diss_coeff,
            rate=fes_diss_coeff * fes_32.value,
            c=c,
        )


def fes_formation_fully_implicit_3(
    c: dict[str, CellVariable],
    k: dict[str, float],
    lim: dict[str, float],
    LHS: dict[str, ImplicitSourceTerm],
    RHS: dict[str, explicitSourceTerm],
    RATES: dict[str, float],
    CROSS: dict[str, list[tuple[str, float]]],
    mp: dict[str, float],
) -> None:
    """
    Assemble the FiPy matrices for the Fe‑S system with **all fast
    precipitation terms treated implicitly**.

    Parameters
    ----------
    c, k, lim, mp
        Dictionaries that hold the FiPy variables, kinetic constants,
        user‑defined limiting factors and model parameters.
    LHS, RHS, RATES, CROSS
        Containers that collect the implicit/explicit source terms,
        diagnostic rate totals and the coupling information required
        later when the global FiPy equation is built.
    """
    # ------------------------------------------------------------------
    # 0️⃣  Short‑hand aliases for readability
    # ------------------------------------------------------------------
    fe2: CellVariable = c["fe2_total"]
    hs: CellVariable = c["ts2"]
    fes: CellVariable = c["fes"]

    # ------------------------------------------------------------------
    # 1️⃣  Non‑linear precipitation rate (fully implicit)
    # ------------------------------------------------------------------
    omega_den = k["hplus"] * k["fes_sp"] + 1e-30
    omega = (fe2 * mp["f_diss"] * hs) / omega_den  # Ω = (Fe²⁺·f_diss·H₂S)/(H⁺·Ksp)
    rate_precip = k["fes_isp"] * (omega - 1.0)  # R = k_isp·(Ω‑1)

    # ------------------------------------------------------------------
    # 2️⃣  Implicit source terms for the three species
    # ------------------------------------------------------------------
    # Fe²⁺  :  -R·f_diss
    term_fe2 = ImplicitSourceTerm(var=fe2, coeff=-rate_precip * mp["f_diss"])
    if "fe2_total" in LHS:
        LHS["fe2_total"].coeff += term_fe2.coeff
    else:
        LHS["fe2_total"] = term_fe2

    # H₂S  :  -R
    term_hs = ImplicitSourceTerm(var=hs, coeff=-rate_precip)
    if "ts2" in LHS:
        LHS["ts2"].coeff += term_hs.coeff
    else:
        LHS["ts2"] = term_hs

    # FeS (solid) :  +R·fac_s
    term_fes = ImplicitSourceTerm(var=fes, coeff=rate_precip * mp["fac_s"])
    if "fes" in LHS:
        LHS["fes"].coeff += term_fes.coeff
    else:
        LHS["fes"] = term_fes

    # ------------------------------------------------------------------
    # 3️⃣  Book‑keeping of the precipitation rate (diagnostics only)
    # ------------------------------------------------------------------
    avg_rate = rate_precip.value
    for sp in ("fe2_total", "ts2", "fes"):
        RATES[sp] = RATES.get(sp, 0.0) + avg_rate

    # ------------------------------------------------------------------
    # 4️⃣  Slow dissolution – still linear, keep the original helpers
    # ------------------------------------------------------------------
    # Ω_diss = min(Ω, 1)  →  (1‑Ω_diss) drives dissolution
    omega_diss = np.minimum(omega, 1.0)
    # mask = 1 where dissolution is active, 0 where precipitation dominates
    diss_mask = 1.0 - np.where(omega > 1.0, 1.0, 0.0)

    fes_diss_coeff = (
        k["fes_isd"] * (1.0 - omega_diss) * diss_mask * lim["fes_explicit"]
    )  # linear coefficient (scalar or array)

    # Fe²⁺ source from dissolution
    add_implicit_coupling(
        CROSS,
        RATES,
        target_species="fe2_total",
        source_species="fes",
        coeff=fes_diss_coeff,
        rate=fes_diss_coeff * fes.value,
        c=c,
    )
    # H₂S source from dissolution
    add_implicit_coupling(
        CROSS,
        RATES,
        target_species="ts2",
        source_species="fes",
        coeff=fes_diss_coeff,
        rate=fes_diss_coeff * fes.value,
        c=c,
    )
    # FeS sink from dissolution (multiply by solid‑phase factor)
    coeff_fes_sink = fes_diss_coeff * mp["fac_s"]
    add_implicit_sink(
        LHS,
        RATES,
        species="fes",
        coeff=coeff_fes_sink,
        rate=coeff_fes_sink * fes.value,
        c=c,
    )

    # ------------------------------------------------------------------
    # 5️⃣  Optional isotope bookkeeping (unchanged – still uses helpers)
    # ------------------------------------------------------------------
    if "ts2_32" in c:
        ts2_32: CellVariable = c["ts2_32"]
        fes_32: CellVariable = c["fes_32"]

        # 5a – precipitation (non‑linear part) for 32S
        ts2_inv = 1.0 / (hs.value + 1e-20)
        coeff_ts2_32_precip = (
            k["fes_isp"]
            * (fe2 * mp["f_diss"] / omega_den - ts2_inv)
            * np.where(omega > 1.0, 1.0, 0.0)
        )
        coeff_ts2_32_precip = np.maximum(coeff_ts2_32_precip, 0.0)

        add_implicit_sink(
            LHS,
            RATES,
            species="ts2_32",
            coeff=coeff_ts2_32_precip,
            rate=coeff_ts2_32_precip * ts2_32.value,
            c=c,
        )

        # 5b – coupling of 32S to the solid
        add_implicit_coupling(
            CROSS,
            RATES,
            target_species="fes_32",
            source_species="ts2_32",
            coeff=coeff_ts2_32_precip * mp["fac_s"],
            rate=coeff_ts2_32_precip * ts2_32.value * mp["fac_s"],
            c=c,
        )

        # 5c – dissolution sink on the solid (same coeff as main solid)
        add_implicit_sink(
            LHS,
            RATES,
            species="fes_32",
            coeff=coeff_fes_sink,
            rate=coeff_fes_sink * fes_32.value,
            c=c,
        )

        # 5d – dissolution source on H₂S_32 (coupled to solid)
        add_implicit_coupling(
            CROSS,
            RATES,
            target_species="ts2_32",
            source_species="fes_32",
            coeff=fes_diss_coeff,
            rate=fes_diss_coeff * fes_32.value,
            c=c,
        )
