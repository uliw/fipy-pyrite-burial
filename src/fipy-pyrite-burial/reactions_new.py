"""Define the reactions."""

from __future__ import annotations

# import numpy as np
from fipy.tools.numerix import *
from fipy.tools import numerix as np
from fipy.variables.cellVariable import CellVariable

from diff_lib import (
    add_explicit_source,
    add_implicit_sink,
    add_implicit_coupling_new,
)


def equilibrium_reactions(mp, c, k, f, RATES, dt):
    """Instantenous reactions.

    that are calculated after the
    transport matrix has been solved.
    """
    for r in mp.instantenous_reactions:
        r(c, k, mp, dt, RATES)

    return f, RATES


def diagenetic_reactions(mp, c, k, f):
    """
    Main orchestrator for diagenetic reactions.  That are inside transport matrix.
    Calculates limiters, initializes matrices, and calls specific process functions.

    Porosity Handling:

    Model units are meter/second, concentrations are given mmol/liter (mol/m^3) and
    solids are expressed as concentration per unit of solid volume (mmol/L_solid).

    This keeps the physics of the "solid phase" independent of how much water is
    currently squeezing around it.  If the sediment compacts (porosity ϕ decreases), the
    amount of organic matter per gram of rock doesn't change, but the amount of organic
    matter per liter of bulk sediment does.  ​

    As such, a reaction between a liquid and a solid needs to be scaled

    f = k * [SO4] * (1 - phi)/phi * [OM]

    This is now handled by the maxtrix helper functions via the ctype parameter, where
    ctype indicates 'rate_phase_2_species_phase'.. Eg.
    when calculting the consumption of organic matter by sulfate you would write

    # POC Sink - SOLID
    coeff_poc = k.poc_o2 * c.so4
    add_implicit_sink(
        LHS, RATES, "poc", coeff_poc, rate_base, ctype="solid", mp=mp, c=c
    )

    whereas the consumption of sulfate would be

    coeff_so4 = k.poc_so4 * c.poc
    add_implicit_sink(LHS, RATES, "so4", coeff_so4, so4_rate, ctype="liquid", mp=mp, c=c)

    or as a coupled reaction:

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

    the porosity correction will then applied automatically depending on the
    ctype paramater,
    """
    from fipy import ImplicitSourceTerm

    # 1. SETUP & INITIALIZATION
    # -------------------------
    species_list = list(c.keys())

    # Accumulators (The State)
    # LHS: Diagonal (Self) Coefficients (Implicit Sinks)
    # LHS = {s: np.zeros_like(c.so4.value) for s in species_list}
    LHS = {s: np.zeros_like(c.so4) for s in species_list}

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

    limiters["ts2"] = 0.1 / (c.ts2 + 0.1)
    limiters["disp_o2_inhib"] = 0.01 / (c.o2 + 0.01)

    # Sulfate Limiter (Implicit 1/[S+K] and Explicit [S]/[S+K])
    K_so4 = 0.2
    limiters["so4_implicit"] = 1.0 / (c.so4 + K_so4)
    limiters["so4_32_implicit"] = 1.0 / (c.so4_32 + K_so4)

    limiters["so4_explicit"] = c.so4 / (c.so4 + K_so4)
    limiters["so4_32_explicit"] = c.so4_32 / (c.so4_32 + K_so4)

    limiters["fe3_explicit"] = c.fe3 / (c.fe3 + 1e-3)
    limiters["fe3_implicit"] = 1.0 / (c.fe3 + 1e-3)

    # limiters["fes_explicit"] = c.fes / (c.fes + 1e-6)
    # limiters["fes_implicit"] = 1 / (c.fes + 1e-6)

    K_alpha = 0.2
    limiters["alpha_explicit"] = c.so4 / (c.so4 + K_alpha)
    limiters["alpha_implicit"] = 1.0 / (c.so4 + K_alpha)

    # H2S Alpha Limiter (prevents numerical issues at trace concentrations)
    limiters["ts2_alpha_explicit"] = c.ts2 / (c.ts2 + 0.05)

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

    for s in species_list:
        lhs_coeff = LHS.get(s, 0.0)
        cross_list = CROSS.get(s, [])

        cross_term = 0.0
        if isinstance(cross_list, list):
            for source_name, coeff in cross_list:
                # Ensure numpy arrays are wrapped in CellVariable for correct rank, but preserve FiPy expressions
                if isinstance(coeff, np.ndarray) and coeff.shape != ():
                    coeff_val = CellVariable(mesh=c[s].mesh, value=coeff)
                else:
                    coeff_val = coeff
                cross_term += ImplicitSourceTerm(coeff=coeff_val, var=c[source_name])

        setattr(f, s, (lhs_coeff, RHS[s], RATES[s], cross_term))

    # print("fe2 LHS (implicit sink coeff):", LHS["fe2_total"].sum())
    # print("fe2 RHS (explicit source):", RHS["fe2_total"].sum())
    # print("fe2 CROSS entries:")
    # for source_name, coeff in CROSS["fe2_total"]:
    #     val = getattr(coeff, "value", coeff)
    #     if hasattr(val, "sum"):
    #         print(f"  from {source_name}: coeff sum = {val.sum()}")
    #     else:
    #         print(f"  from {source_name}: coeff = {val}")
    #         print("fe2 RATES:", RATES["fe2_total"].sum())
    # print()

    return f, RATES


# =============================================================================
# PROCESS FUNCTIONS (The Biogeochemistry)
# =============================================================================
def aerobic_respiration(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define POC consumption by aerobic respiration."""
    rate_base = k.poc_o2 * c.poc * c.o2

    # POC Sink - SOLID
    coeff_poc = k.poc_o2 * c.o2
    add_implicit_sink(
        LHS, RATES, "poc", coeff_poc, rate_base, ctype="solid", mp=mp, c=c
    )

    # O2 Sink (1.27x) - LIQUID
    coeff_o2 = 1.27 * k.poc_o2 * c.poc
    add_implicit_sink(
        LHS, RATES, "o2", coeff_o2, 1.27 * rate_base, ctype="liquid", mp=mp, c=c
    )
    # No produced species here (CO2 ignored)


def sulfate_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate sulfate reduction.

    Reaction: 2 POC + 1 SO4 -> 1 TS2- Ref: POC (k.poc_so4)
    """
    # 1. Base Rate
    poc_rate = k.poc_so4 * c.poc * c.so4 * lim["so4_implicit"] * lim["inhib_o2"]
    so4_rate = poc_rate * 0.5

    # 2. POC Sink (Ref Species) - SOLID
    coeff_poc = k.poc_so4 * lim["so4_explicit"] * lim["inhib_o2"]
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, poc_rate, ctype="solid", mp=mp, c=c)

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
    if mp.isotopes:
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
    add_implicit_sink(
        LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2, ctype="liquid", mp=mp, c=c
    )

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

    if mp.isotopes:
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
    Reaction: 1 S0 + 1.5 O2 -> 1 SO4
    Assuming that some O comes from H2O
    Phases: S0 (Solid), O2 (Liquid), SO4 (Liquid)
    """
    # S0 sink (Solid)
    # Rate = k * [O2] * [S0]
    coeff_s0 = k.s0_ox * c.o2

    # O2 Sink (1.5x) - LIQUID
    coeff_o2 = 1.5 * k.s0_ox * c.s0
    add_implicit_sink(
        LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2, ctype="solid_2_liquid", mp=mp, c=c
    )

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

    if mp.isotopes:
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

    0.5 HS- + Fe3+ -> 0.5 S0 + Fe2+

    The reaction is doubly-capped so that at most 70% of either fe3 or ts2
    is consumed in a single timestep, guaranteeing perfect stoichiometry and
    unconditional positivity. Coupling uses a single master implicit variable
    (fe3) to ensure exact mass balance across all species.
    """
    # ------------------------------------------------------------------
    # 1. Current state (use .value to avoid large expression trees)
    # ------------------------------------------------------------------
    ts2_val = np.maximum(getattr(c.ts2, "value", c.ts2), 0.0)
    hs_frac_val = getattr(mp.hs_frac, "value", mp.hs_frac)
    hs_val = ts2_val * hs_frac_val
    fe3_val = np.maximum(getattr(c.fe3, "value", c.fe3), 0.0)

    # ------------------------------------------------------------------
    # 2. Rate Calculation and Multi-Reactant Capping
    # ------------------------------------------------------------------
    k_base_val = (
        getattr(k.fe3_hs, "value", k.fe3_hs)
        * getattr(lim["inhib_o2"], "value", lim["inhib_o2"])
        * getattr(lim["fe3_implicit"], "value", lim["fe3_implicit"])
    )

    # Uncapped reaction rate driven by fe3 consumption [mmol/L_solid/s]
    rate_uncapped = k_base_val * hs_val * fe3_val

    # Limit by fe3 depletion (at most 70% per timestep)
    max_rate_fe3 = 0.7 * fe3_val / (mp.current_dt + 1e-30)

    # Limit by ts2 depletion (0.5 mole ts2 consumed per 1 mole fe3)
    # 0.5 * Rate * dt <= 0.7 * ts2 -> Rate <= 1.4 * ts2 / dt
    max_rate_ts2 = 1.4 * ts2_val / (mp.current_dt + 1e-30)

    # Actual capped rate
    rate_actual = np.minimum(rate_uncapped, np.minimum(max_rate_fe3, max_rate_ts2))

    # Single master coefficient based on fe3 [1/s]
    coeff_master = rate_actual / (fe3_val + 1e-30)

    # ------------------------------------------------------------------
    # 3. Fe3 sink (Master Variable) — EXACTLY 1:1
    # ------------------------------------------------------------------
    add_implicit_sink(
        LHS, RATES, "fe3", coeff_master, rate_actual, ctype="solid", mp=mp, c=c
    )

    # ------------------------------------------------------------------
    # 4. Fe2 source (Coupled to fe3_new) — EXACTLY 1:1
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        "solid_2_liquid",
        CROSS,
        RATES,
        LHS,
        target_species="fe2_total",
        source_species="fe3",
        coeff=coeff_master,
        rate=rate_actual,
        mp=mp,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=1.0,
    )

    # ------------------------------------------------------------------
    # 5. ts2 sink (Coupled to fe3_new) — EXACTLY 0.5:1
    # ------------------------------------------------------------------
    # fe3 is the source_species in the CROSS coupling, meaning ts2 receives
    # -0.5 * coeff_master * fe3_new on its RHS.
    CROSS["ts2"].append(("fe3", -0.5 * coeff_master))
    RATES["ts2"] -= 0.5 * rate_actual

    # ------------------------------------------------------------------
    # 6. s0 source (Coupled to fe3_new) — EXACTLY 0.5:1
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        "solid_2_solid",
        CROSS,
        RATES,
        LHS,
        target_species="s0",
        source_species="fe3",
        coeff=coeff_master * 0.5,
        rate=0.5 * rate_actual,
        mp=mp,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=1.0,
    )

    # ------------------------------------------------------------------
    # 7. Isotopes (32S)
    # ------------------------------------------------------------------
    if mp.isotopes:
        ts2_32_val = getattr(c.ts2_32, "value", c.ts2_32)
        f32 = ts2_32_val / (ts2_val + 1e-30)

        rate_32 = 0.5 * rate_actual * f32
        coeff_32 = 0.5 * coeff_master * f32

        # ts2_32 sink
        CROSS["ts2_32"].append(("fe3", -coeff_32))
        RATES["ts2_32"] -= rate_32

        # s0_32 source
        add_implicit_coupling_new(
            "solid_2_solid",
            CROSS,
            RATES,
            LHS,
            target_species="s0_32",
            source_species="fe3",
            coeff=coeff_32,
            rate=rate_32,
            mp=mp,
            c=c,
            add_lhs_sink=False,
            stoich_ratio=1.0,
        )


def sulfide_speciation_clip(c, k, mp, dt, RATES):
    """Update reporting species (h2s, hs) based on total sulfide (ts2) and pH."""
    ts2_val = c.ts2.value
    c.h2s.value[:] = ts2_val * mp.h2s_frac
    c.hs.value[:] = ts2_val * mp.hs_frac

    if mp.isotopes:
        ts2_32_val = c.ts2_32.value
        c.h2s_32.value[:] = ts2_32_val * mp.h2s_frac
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

    # O2 Sink (1/4) - LIQUID
    coeff_o2 = k.fe2_ox * c.fe2_total * 0.25
    add_implicit_sink(
        LHS,
        RATES,
        "o2",
        coeff_o2,
        rate_base * 0.25,
        ctype="liquid",
        mp=mp,
        c=c,
    )

    # Fe3 Source (1.0x) - SOLID, coupled to Fe2_total (liquid)
    add_implicit_coupling_new(
        "liquid_2_solid",
        CROSS,
        RATES,
        LHS,
        "fe3",  # product (solid)
        "fe2_total",  # source (liquid)
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
    add_implicit_sink(
        LHS,
        RATES,
        "o2",
        coeff_o2_fes,
        rate_base * 2.25,
        ctype="solid_2_liquid",
        mp=mp,
        c=c,
    )

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

    if mp.isotopes:
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
    # S0 Sink - SOLID
    coeff_s0 = k.fes_s0 * c.fes
    add_implicit_sink(
        LHS, RATES, "s0", coeff_s0, coeff_s0 * c.s0, ctype="solid", mp=mp, c=c
    )

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

    if mp.isotopes:
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
    add_implicit_sink(
        LHS,
        RATES,
        "ts2",
        coeff_ts2,
        coeff_ts2 * c.ts2,
        ctype="solid_2_liquid",
        mp=mp,
        c=c,
    )

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

    if mp.isotopes:
        # 1. Isotope sinks (on species that are consumed)
        # H2S_32 (liquid) sink
        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            coeff_ts2,
            coeff_ts2 * c.ts2_32,
            ctype="solid_2_liquid",
            mp=mp,
            c=c,
        )

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
    raise DeprecationWarning()
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
    if mp.isotopes:
        # No fractionation during pyrite oxidation: consume fes2_32
        # proportionally to its abundance.  Use the same coefficient as the
        # bulk (coeff_fes2), so the implicit sink is coeff_fes2 * fes2_32
        # (linear).  Multiplying by f32_fes2 first would make it quadratic
        # in fes2_32 and corrupt the remaining δ³⁴S.
        add_implicit_coupling_new(
            "solid_2_liquid",
            CROSS,
            RATES,
            LHS,
            target_species="so4_32",
            source_species="fes2_32",
            coeff=coeff_fes2,  # same as bulk — keeps sink linear
            rate=coeff_fes2 * c.fes2_32,  # explicit rate uses _32 variable
            mp=mp,
            c=c,
            stoich_ratio=1.0,  # fes2_32 in mol-S, not mol-FeS2
        )


def fes_precipitation_clip(c, k, mp, dt, RATES):
    """Fe2+ HS- -> FeS
    Instantaneous equilibrium 'Clip' for FeS precipitation.
    Includes Solid-Phase Buffering!
    """
    #     import numpy as np

    # 1. Get concentrations
    fe_pw = c.fe2_total.value * mp.f_diss
    ts2_pw = c.ts2.value
    fe_pw = c.fe2_total * mp.f_diss
    ts2_pw = c.ts2

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

    phi_val = getattr(mp.phi, "value", mp.phi)
    fac_s = phi_val / (1.0 - phi_val)

    # --- REPORTING LOGIC ---
    rate_report = x_precip / dt

    RATES["ts2"][mask] -= rate_report
    RATES["fe2_total"][mask] -= rate_report
    RATES["fes"][mask] += rate_report * fac_s[mask]

    # ---------------------------------------------------------
    # 5. Calculate Isotope Mass Transfer FIRST
    # ---------------------------------------------------------
    if mp.isotopes:
        # Current Porewater Ratio
        R_pw = c.ts2_32.value[mask] / (c.ts2.value[mask] + 1e-20)

        # Mass of 32S we WANT to move
        loss_32 = x_precip * R_pw

        # Apply changes
        c.ts2_32.value[mask] -= loss_32

        # Ensure your solid isotope variable matches your initialization
        c.fes_32.value[mask] += loss_32 * mp.fac_s

    # ---------------------------------------------------------
    # 6. Update Bulk State Variables
    # ---------------------------------------------------------
    c.ts2.value[mask] -= x_precip
    c.fe2_total.value[mask] -= x_precip
    c.fes.value[mask] += x_precip * fac_s[mask]

    return RATES


def fes_precipitation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    FeS precipitation — ts2-primary, quadratic-bounded.

    Uses v9's proven CROSS structure (ts2 drives fe2 and fes) with
    a quadratic-bounded target that replaces fe2_limiter.

    Equilibrium (v9 convention, 1:1 stoich for fe2:ts2):
      fe2 · ts2 · hs_frac = omega_den
      K_eff = omega_den / hs_frac

    Quadratic bound — max reaction progress X where both deplete:
      (fe2 - X)(ts2 - X) = K_eff
      X_max = ½[(fe2 + ts2) - √((fe2 - ts2)² + 4·K_eff)]

    Properties:
      X_max ≤ min(fe2, ts2)     → neither overconsumes
      X_max → 0 as fe2 → 0     → no fe2-depletion oscillation
      X_max = 0 when Ω ≤ 1     → no precipitation when undersaturated

    ts2_target = ts2 - X_max   → precipitation limit encoded in target

    Equation structure (identical to v9):
      ts2:  self-implicit sink  + explicit stabilisation at ts2_target
      fe2:  CROSS to ts2_new   (same coeff as ts2 sink, 1:1 stoich)
      fes:  CROSS to ts2_new   (same coeff, opposite sign, NO fac_s)
            eff_phi handles phase-volume conversion automatically

    Conservation: all three driven by same ts2_new → exact per timestep.
    """
    # ------------------------------------------------------------------
    # 1. Current state
    # ------------------------------------------------------------------
    # fe2_pw_val = c.fe2_total.value * mp.f_diss + 1e-20
    # ts2_val = c.ts2.value
    # hs_val = ts2_val * mp.hs_frac
    # fes_val = c.fes.value
    fe2_pw_val = c.fe2_total * mp.f_diss + 1e-20
    ts2_val = c.ts2
    hs_val = ts2_val * mp.hs_frac
    fes_val = c.fes

    # ------------------------------------------------------------------
    # 2. Quadratic bound
    #
    #   (fe2 - X)(ts2 - X) = K_eff
    #   X_max = ½[(fe2+ts2) - √((fe2-ts2)² + 4·K_eff)]
    # ------------------------------------------------------------------
    omega_den = k.hplus * k.fes_sp + 1e-30
    K_eff = omega_den / (mp.hs_frac + 1e-30)
    discriminant = (fe2_pw_val - ts2_val) ** 2 + 4.0 * K_eff
    X_max = 0.5 * ((fe2_pw_val + ts2_val) - np.sqrt(discriminant))
    X_max = np.maximum(X_max, 0.0)

    ts2_target = ts2_val - X_max

    # ------------------------------------------------------------------
    # 3. Regime flag and rate coefficient
    #
    #   k_rxn includes hs_frac (v9 convention): rate scales with HS
    #   availability.  No fe2_limiter needed — X_max bounds the target.
    # ------------------------------------------------------------------
    omega = (fe2_pw_val * hs_val) / omega_den
    sharpness = 100.0
    is_precip = 0.5 * (1.0 + np.tanh(sharpness * (omega - 1.0)))

    k_rxn = k.fes_isp * is_precip * mp.hs_frac  # [1/s, L_pw]

    # ------------------------------------------------------------------
    # 4. Reporting
    # ------------------------------------------------------------------
    precip_rate = k_rxn * np.maximum(ts2_val - ts2_target, 0.0)

    # ------------------------------------------------------------------
    # 5a. ts2 — self-implicit relaxation toward ts2_target
    #
    #   d[ts2]/dt = -k_rxn·ts2 + k_rxn·ts2_target
    #   ts2_new → weighted avg of ts2_old and ts2_target → bounded
    # ------------------------------------------------------------------
    add_implicit_sink(LHS, RATES, "ts2", k_rxn, precip_rate, ctype="liquid")
    add_explicit_source(
        RHS, RATES, "ts2", k_rxn * ts2_target, update_rates=False, ctype="liquid"
    )

    # ------------------------------------------------------------------
    # 5b. fe2_total — CROSS to ts2_new (v9 convention)
    #
    #   Same coefficient as ts2 sink → 1:1 stoichiometry
    #   Δfe2 = Δts2 (both deplete by same amount)
    # ------------------------------------------------------------------
    CROSS["fe2_total"].append(("ts2", -k_rxn))
    # RATES["fe2_total"] -= getattr(precip_rate, "value", precip_rate)
    RATES["fe2_total"] -= precip_rate

    add_explicit_source(
        RHS, RATES, "fe2_total", k_rxn * ts2_target, update_rates=False, ctype="liquid"
    )

    # ------------------------------------------------------------------
    # 5c. fes — CROSS to ts2_new (v9 convention)
    #
    #   Same coefficient, opposite sign.  NO fac_s!
    #   eff_phi on transient term handles phase-volume conversion.
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        "liquid_2_solid",
        CROSS,
        RATES,
        LHS,
        target_species="fes",
        source_species="ts2",
        coeff=k_rxn,
        rate=precip_rate,
        mp=mp,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=1.0,
    )
    add_explicit_source(
        RHS, RATES, "fes", -k_rxn * ts2_target, update_rates=False, ctype="solid"
    )

    # ------------------------------------------------------------------
    # 6. Isotopes (32S)
    # ------------------------------------------------------------------
    if mp.isotopes:
        # ts2_32_val = c.ts2_32.value
        ts2_32_val = c.ts2_32
        f32_ts2 = ts2_32_val / (ts2_val + 1e-30)

        ts2_32_target = ts2_target * f32_ts2
        precip_rate_32 = k_rxn * np.maximum(ts2_32_val - ts2_32_target, 0.0)

        # ts2_32: self-implicit
        add_implicit_sink(LHS, RATES, "ts2_32", k_rxn, precip_rate_32, ctype="liquid")
        add_explicit_source(
            RHS,
            RATES,
            "ts2_32",
            k_rxn * ts2_32_target,
            update_rates=False,
            ctype="liquid",
        )

        # fes_32: CROSS to ts2_32
        add_implicit_coupling_new(
            "liquid_2_solid",
            CROSS,
            RATES,
            LHS,
            target_species="fes_32",
            source_species="ts2_32",
            coeff=k_rxn,
            rate=precip_rate_32,
            mp=mp,
            c=c,
            add_lhs_sink=False,
        )
        add_explicit_source(
            RHS,
            RATES,
            "fes_32",
            -k_rxn * ts2_32_target,
            update_rates=False,
            ctype="solid",
        )


def fes_precipitation_terminal(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    FeS precipitation as terminal solid phase. No dissolution.
    Fe2 + HS- -> FeS

    All bulk species and fes_32 driven by fe2_new.
    ts2_32 has its own self-implicit sink with the same coefficient
    as bulk ts2, preserving the isotope ratio during depletion.
    """
    fe2_pw_val = c.fe2_total * mp.f_diss + 1e-20
    hs_val = c.ts2 * mp.hs_frac
    fes_coeff = k.fes_ts2 * hs_val
    hs_coeff = k.fes_ts2 * fe2_pw_val

    # Fe2 sink + FeS source: CROSS to fe2_new
    add_implicit_coupling_new(
        "liquid_2_solid",
        CROSS,
        RATES,
        LHS,
        target_species="fes",
        source_species="fe2_total",
        coeff=fes_coeff,
        rate=fes_coeff * fe2_pw_val,
        mp=mp,
        c=c,
        add_lhs_sink=True,
        stoich_ratio=1.0,
    )

    # TS2 sink: self-implicit
    add_implicit_sink(LHS, RATES, "ts2", hs_coeff, hs_coeff * hs_val, ctype="liquid")

    if mp.isotopes:
        f32_ts2 = c.ts2_32 / (c.ts2 + 1e-30)

        # fes_32: CROSS to fe2_new, scaled by porewater isotope ratio
        # Driven by same fe2_new as bulk fes → fes_32/fes = f32_ts2
        add_implicit_coupling_new(
            "liquid_2_solid",
            CROSS,
            RATES,
            LHS,
            target_species="fes_32",
            source_species="fe2_total",
            coeff=fes_coeff * f32_ts2,
            rate=fes_coeff * fe2_pw_val * f32_ts2,
            mp=mp,
            c=c,
            add_lhs_sink=False,  # fe2 sink already registered above
        )

        # ts2_32: self-implicit sink, same coeff as bulk ts2
        # ts2_32/ts2 ratio preserved because same coefficient
        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            hs_coeff,
            hs_coeff * c.ts2_32,
            ctype="liquid",
        )


def fes_dissolution(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    FeS dissolution — equilibrium-capped, hard regime gate.

    All rates in natural model units (mmol/L_pw or mmol/L_solid).
    Porosity conversions are handled transparently by helpers via ctype.

    Hard cutoff at omega < omega_crit prevents any overlap with the
    precipitation function.
    """
    from fipy.variables.variable import Variable

    # ------------------------------------------------------------------
    # 1. Current state
    # ------------------------------------------------------------------
    fe2_pw_val = c.fe2_total * mp.f_diss + 1e-20  # mmol/L_pw
    ts2_val = c.ts2  # mmol/L_pw
    hs_val = ts2_val * mp.hs_frac  # mmol/L_pw
    fes_val = c.fes  # mmol/L_solid
    fes_limiter = fes_val / (fes_val + 1e-4)

    # ------------------------------------------------------------------
    # 2. Regime gate — hard cutoff, no overlap with precipitation
    # ------------------------------------------------------------------
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_pw_val * hs_val) / omega_den
    omega_crit = 0.95
    is_diss = np.where(omega < omega_crit, 1.0, 0.0)

    # ------------------------------------------------------------------
    # 3. Dissolution coefficient — double cap
    # ------------------------------------------------------------------
    k_d_raw = k.fes_isd * fes_limiter * is_diss  # [1/s, L_solid basis]

    # Cap 1: reservoir — dissolve at most f_max of FeS per timestep
    f_max = 0.5
    # quick and dirty to get cell variable
    k_d_reservoir = k_d_raw * 0 + f_max / (mp.current_dt + 1e-30)
    # k_d_reservoir = Variable(f_max / (mp.current_dt + 1e-30))

    # Cap 2: equilibrium — products cannot push omega above omega_crit.
    #   Solve for Y [mmol/L_pw]: (fe2_pw + Y)(hs + Y*hs_frac) = target_product
    #   Y*hs_frac*Y + (fe2_pw*hs_frac + hs)*Y + (fe2_pw*hs - target) = 0
    target_product = omega_crit * omega_den
    deficit = np.maximum(target_product - fe2_pw_val * hs_val, 0.0)
    B = fe2_pw_val * mp.hs_frac + hs_val
    discriminant = B**2 + 4.0 * mp.hs_frac * deficit
    Y_max = np.maximum(
        (-B + np.sqrt(discriminant + 1e-30)) / (2.0 * mp.hs_frac + 1e-30), 0.0
    )

    # Y_max is in mmol/L_pw; convert to mmol/L_solid via fac_s = (1-phi)/phi,
    # then divide by fes to get a [1/s] coefficient.
    # fac_s converts: mmol/L_pw * fac_s = mmol/L_solid
    k_d_equil = Y_max * mp.fac_s / (fes_val + 1e-30) / (mp.current_dt + 1e-30)

    # Apply both caps
    k_d = np.minimum(k_d_raw, np.minimum(k_d_reservoir, k_d_equil))

    # ------------------------------------------------------------------
    # 4. Dissolution rate [mmol/L_solid/s]
    # ------------------------------------------------------------------
    diss_rate_solid = k_d * fes_val

    # print(f"f max = {np.max(diss_rate_solid.value):.2e}, f min = {np.min(diss_rate_solid.value):.2e}")

    # ------------------------------------------------------------------
    # 5a. fes sink + ts2 source
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        "solid_2_liquid",
        CROSS,
        RATES,
        LHS,
        target_species="ts2",
        source_species="fes",
        coeff=k_d,
        rate=diss_rate_solid,
        mp=mp,
        c=c,
        add_lhs_sink=True,
    )

    # ------------------------------------------------------------------
    # 5b. fe2_total source (solid_2_liquid; fe2_total is a diffusing liquid)
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        "solid_2_liquid",
        CROSS,
        RATES,
        LHS,
        target_species="fe2_total",
        source_species="fes",
        coeff=k_d,
        rate=diss_rate_solid,
        mp=mp,
        c=c,
        add_lhs_sink=False,
    )

    # ------------------------------------------------------------------
    # 6. Isotopes (32S) — no fractionation, k_d identical to bulk
    # ------------------------------------------------------------------
    if mp.isotopes:
        fes_32_val = c.fes_32
        diss_32_solid = k_d * fes_32_val

        add_implicit_coupling_new(
            "solid_2_liquid",
            CROSS,
            RATES,
            LHS,
            target_species="ts2_32",
            source_species="fes_32",
            coeff=k_d,
            rate=diss_32_solid,
            mp=mp,
            c=c,
            add_lhs_sink=True,
        )



def s0_disproportionation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate elemental sulfur disproportionation.

    Reaction: 4S0 + 4.5O2 -> 1H2S + 3SO4

    Notes:
    - The split between H2S and SO4 depends on mp.dispro_so4_hs_split,
      which is the ratio between H2S/SO4, typically 1:2 -> 0.5
    - The isotope fractionation between S0 and H2S is given by mp.dispro_hs_alpha (0.993)
    - The isotope fractionation between S0 and SO4 is given by mp.dispro_so4_alpha (1.02)
    - The reaction constant for the overall reaction is given by k.s0_dispro
    - The reaction rate depends on S0, and H2S & O2 as inhibitors.
    """

    # 1. Base Rate Calculation (Master Species: S0)
    # Disproportionation is anaerobic, so O2 is NOT a reactant.
    # Instead, we use the inhibitor lim["disp_o2_inhib"] to ensure it only
    # proceeds under low oxygen conditions.
    rate_uncapped = k.s0_dispro * c.s0 * lim["ts2"] * lim["disp_o2_inhib"]

    # Capping to prevent over-consumption in a single timestep
    max_rate_s0 = 0.7 * c.s0 / (mp.current_dt + 1e-30)
    rate_actual = np.minimum(rate_uncapped, max_rate_s0)

    # Coefficient based on the consumed species (S0)
    coeff_s0_base = rate_actual / (c.s0 + 1e-30)

    # 2. Calculate the Stoichiometric Split
    # If split = 0.5 (1 H2S : 2 SO4), then for 1.5 moles of S0:
    # 1.0 mole goes to SO4 and 0.5 moles go to H2S
    split = mp.dispro_so4_hs_split
    so4_fraction = 1.0 / (1.0 + split)
    h2s_fraction = split / (1.0 + split)

    # SO4 Production
    coeff_so4 = coeff_s0_base * so4_fraction
    add_implicit_coupling_new(
        "solid_2_liquid",
        CROSS,
        RATES,
        LHS,
        "so4",
        "s0",
        coeff_so4,
        coeff_so4 * c.s0,
        mp,
        c=c,
        add_lhs_sink=True,
    )

    # TS2 (H2S) Production
    coeff_ts2 = coeff_s0_base * h2s_fraction
    add_implicit_coupling_new(
        "solid_2_liquid",
        CROSS,
        RATES,
        LHS,
        "ts2",
        "s0",
        coeff_ts2,
        coeff_ts2 * c.s0,
        mp,
        c=c,
        add_lhs_sink=True,
    )

    # O2 Consumption - REMOVED (Disproportionation is anaerobic)
    # If the user intended this to be S0 oxidation, O2 should be a reactant.
    # But for disproportionation, it is purely internal redox.

    # 5. Isotopes
    if mp.isotopes:
        # To maintain isotope mass balance, the 32S leaving S0 must exactly equal
        # the 32S entering H2S and SO4. If the user-provided alphas do not have a 
        # weighted average of 1.0, mass is created/destroyed. We normalize them here:
        weighted_alpha = h2s_fraction * mp.dispro_hs_alpha + so4_fraction * mp.dispro_so4_alpha
        norm_hs_alpha = mp.dispro_hs_alpha / weighted_alpha
        norm_so4_alpha = mp.dispro_so4_alpha / weighted_alpha

        # Fractionation for H2S path
        coeff_hs_32 = coeff_ts2 * norm_hs_alpha

        add_implicit_coupling_new(
            "solid_2_liquid",
            CROSS,
            RATES,
            LHS,
            "ts2_32",
            "s0_32",
            coeff_hs_32,
            coeff_hs_32 * c.s0_32,
            mp,
            c=c,
            add_lhs_sink=True,
        )

        # Fractionation for SO4 path
        coeff_so4_32 = coeff_so4 * norm_so4_alpha

        add_implicit_coupling_new(
            "solid_2_liquid",
            CROSS,
            RATES,
            LHS,
            "so4_32",
            "s0_32",
            coeff_so4_32,
            coeff_so4_32 * c.s0_32,
            mp,
            c=c,
            add_lhs_sink=True,
        )
