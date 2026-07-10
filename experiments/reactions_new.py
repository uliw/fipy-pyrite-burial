"""Define the reactions.

Note on numerix (nx):
We use `fipy.tools.numerix` instead of standard `numpy` because the reaction functions
are called polymorphically:
1) During setup/plotting, they are called with FiPy `CellVariable`s to build lazy-evaluated equation trees.
2) During time-stepping, they are called with `ArrayProxy` wrapping raw NumPy arrays for high speed.
Using `nx` allows these mathematical operations (e.g. `nx.maximum`, `nx.sqrt`) to build lazy expression
trees in (1) and run at full NumPy speeds in (2). Standard `numpy` would force immediate evaluation in (1).
"""

from __future__ import annotations  # noqa: I001

from fipy.tools import numerix as nx

from fipy.variables.cellVariable import CellVariable

from fipyrite.diff_lib import (
    add_coupled_reaction,
    add_explicit_source,
    add_implicit_coupling_new,
    add_implicit_sink,
    calculate_fractionated_coeff_32,
    partition_equilibrium_isotope_32,
)


def equilibrium_reactions(mp, c, k, f, RATES, dt):
    """Instantenous reactions.

    that are calculated after the
    transport matrix has been solved.
    """
    import inspect

    for r in mp.instantenous_reactions:
        sig = inspect.signature(r[0])
        if "f" in sig.parameters:
            r[0](c, r[1], mp, dt, RATES, f=f)
        else:
            r[0](c, r[1], mp, dt, RATES)

    return f, RATES


def diagenetic_reactions(mp, c, k, f):
    """
    Orchestrate diagenetic reactions inside the transport matrix.

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

    This is now handled by the maxtrix helper functions via the has_solids parameter, where
    has_solid indicates weather the reaction inolves solids. E.g.,
    when calculting the consumption of organic matter by sulfate you would write

    # POC Sink - SOLID
    coeff_poc = k.poc_o2 * c.so4
    add_implicit_sink(
        LHS, RATES, "poc", coeff_poc, rate_base, has_solid=has_solid, mp=mp, c=c
    )

    whereas the consumption of sulfate would be

    coeff_so4 = k.poc_so4 * c.poc
    add_implicit_sink(LHS, RATES, "so4", coeff_so4, so4_rate, has_solid=has_solid, mp=mp, c=c)

    or as a coupled reaction:

    add_implicit_coupling_new(
        CROSS,  #  Off-diagonal coupling matrix
        RATES,  #  Rate reporting dictionary
        LHS,  # Diagonal matrix (implicit sinks)
        "ts2",  # species that is produced
        "so4",  # species that is consumed
        coeff_so4,  # reaction coefficient
        so4_rate,  # coeff * concentration
        mp=mp,
        has_solid= has_solid # type
        c=c,  # model parameters
    )

    the porosity correction will then applied automatically depending on the
    has_solid parameter,
    """
    from fipy import ImplicitSourceTerm

    # 1. SETUP & INITIALIZATION
    # -------------------------
    species_list = list(c.keys())

    # Accumulators (The State)
    # LHS: Diagonal (Self) Coefficients (Implicit Sinks)
    # LHS = {s: nx.zeros_like(c.so4.value) for s in species_list}
    LHS = {s: nx.zeros_like(c.so4) for s in species_list}

    # CROSS: Off-Diagonal / Coupled Terms
    CROSS = {s: [] for s in species_list}

    RHS = {s: nx.zeros_like(c.so4) for s in species_list}
    RATES = {s: nx.zeros_like(c.so4) for s in species_list}

    # 2. CALCULATE LIMITERS
    # ---------------------
    limiters = {}

    # Velde et al specify their k-values in bulk concentrations, so
    # we convert to phase specific space first
    phi = mp.phi
    K_o2 = mp.K_o2 / phi
    K_ts2 = mp.K_ts2 / phi
    K_so4 = mp.K_so4 / phi
    K_fe3 = mp.K_fe3 / (1 - phi)
    K_fe3_diss_red = mp.K_fe3_diss_red / (1 - phi)

    # O2 Limitation for low O2 (1.0 -> 0.0)
    limiters["o2_implicit"] = 1 / (c.o2 + K_o2)
    # O2 Inhibition for high O2 (1.0 -> 0.0)
    limiters["o2_inhibit"] = K_o2 / (c.o2 + K_o2)

    # TS2 inhibition for high TS2
    limiters["ts2"] = K_ts2 / (c.ts2 + K_ts2)

    # Sulfate Limiter (Implicit 1/[S+K] and Explicit [S]/[S+K])
    limiters["so4_implicit"] = 1.0 / (c.so4 + K_so4)
    limiters["so4_explicit"] = c.so4 / (c.so4 + K_so4)

    # isotope enrichment during MSR
    limiters["so4_alpha_explicit"] = c.so4 / (c.so4 + mp.K_epsilon_msr)
    # isotope enrichment during HS oxidation
    limiters["ts2_alpha_explicit"] = c.ts2 / (c.ts2 + mp.K_epsilon_hs_ox)

    # Fe3 limiters
    limiters["fe3_implicit"] = 1.0 / (c.fe3 + K_fe3)
    limiters["fe3_diss_red_implicit"] = 1.0 / (c.fe3 + K_fe3_diss_red)
    limiters["fe3_diss_red_inhib"] = K_fe3_diss_red / (c.fe3 + K_fe3_diss_red)

    # 3. RUN PROCESSES
    # ----------------
    # Each function updates LHS, RHS, and RATES in place
    # r is list of list where r[0] is the function and r[1] is the k val

    for r in mp.diagenetic_reactions:
        r[0](c, r[1], limiters, LHS, RHS, RATES, CROSS, mp)

    for s in species_list:
        lhs_coeff = LHS.get(s, 0.0)
        cross_list = CROSS.get(s, [])

        cross_term = None

        setattr(f, s, (lhs_coeff, RHS[s], RATES[s], cross_term))

    # Pack process-specific rates dynamically
    for key, val in RATES.items():
        if key not in species_list:
            setattr(f, key, (None, None, val, None))

    # Store raw accumulators for static allocation (Option C)
    object.__setattr__(f, "raw_LHS", LHS)
    object.__setattr__(f, "raw_CROSS", CROSS)
    object.__setattr__(f, "raw_RHS", RHS)

    return f, RATES


# =============================================================================
# PROCESS FUNCTIONS (The Biogeochemistry)
# =============================================================================
def aerobic_respiration(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define POC and O2 consumption by aerobic respiration.

    f = k * [OM] * [O2] / (K_O2 *[O2])

    The model does however not track HCO3
    This function considers two types of organic matter
    """
    has_solid = True  # True if the reaction involves solids
    # POC Sink - SOLID
    coeff_poc_fast = k.poc_fast * c.o2 * lim["o2_implicit"]
    coeff_poc_slow = k.poc_slow * c.o2 * lim["o2_implicit"]
    add_implicit_sink(
        LHS,
        RATES,
        "poc_fast",
        coeff_poc_fast,
        coeff_poc_fast * c.poc_fast,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )
    add_implicit_sink(
        LHS,
        RATES,
        "poc_slow",
        coeff_poc_slow,
        coeff_poc_slow * c.poc_slow,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # O2 Sink (1.27x) - LIQUID
    coeff_o2_fast = k.poc_fast * c.poc_fast
    coeff_o2_slow = k.poc_slow * c.poc_slow
    coeff_o2 = (
        (coeff_o2_fast + coeff_o2_slow) * lim["o2_implicit"] * mp.om_o2_consumption
    )
    add_implicit_sink(
        LHS,
        RATES,
        "o2",
        coeff_o2,
        coeff_o2 * c.o2,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # No produced species here (CO2 ignored)


def dissimilatory_iron_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate dissimilatory iron reduction.

    Reaction: CH2O + 4 Fe3 -> HCO3 + 4Fe2+

    This reaction is inhibited under oxic condition, and uses
    a Monod type limiter for low Fe3 concentrations
    """
    has_solid = True  # True if the reactants contain a solid phase species
    inhibit = lim["o2_inhibit"] * lim["fe3_diss_red_implicit"]
    coeff_fe3_slow = k.poc_slow * c.poc_slow
    coeff_fe3_fast = k.poc_fast * c.poc_fast
    coeff_fe3 = 4 * (coeff_fe3_slow + coeff_fe3_fast) * inhibit

    # Couple Fe3 reduction to Fe2 production
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="fe3",
        reactants={},
        products={"fe2_total": 1.0},
        coeff_master=coeff_fe3,
        rate_master=coeff_fe3 * c.fe3,
        has_solid=has_solid,
        reaction_name="dissimilatory_iron_reduction",
    )

    # create the OM sinks
    coeff_poc_slow = k.poc_slow * c.fe3 * inhibit
    coeff_poc_fast = k.poc_fast * c.fe3 * inhibit

    add_implicit_sink(
        LHS,
        RATES,
        "poc_slow",
        coeff_poc_slow,
        coeff_poc_slow * c.poc_slow,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )
    add_implicit_sink(
        LHS,
        RATES,
        "poc_fast",
        coeff_poc_fast,
        coeff_poc_fast * c.poc_fast,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )


def sulfate_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate sulfate reduction.

    Reaction: 2 POC + 1 SO4 -> 1 TS2-

    Notes
    -----
     - sulfate reduction is limited in the presence of O2
     - POC can be fast and/or slow
    """
    has_solid = True  # True if the reactants contain a solid phase species
    inhibition = lim["o2_inhibit"] * lim["so4_implicit"] * lim["fe3_diss_red_inhib"]
    coeff_so4_slow = k.poc_slow * c.poc_slow
    coeff_so4_fast = k.poc_fast * c.poc_fast
    coeff_so4 = (coeff_so4_fast + coeff_so4_slow) * inhibition * 0.5

    # 4. Couple sulfate reduction to h2s production
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="so4",
        reactants={},
        products={"ts2": 1.0},
        coeff_master=coeff_so4,
        rate_master=coeff_so4 * c.so4,
        has_solid=has_solid,
        reaction_name="sulfate_reduction",
    )

    # sulfate reduction consumes poc
    coeff_poc_slow = k.poc_slow * c.so4 * inhibition
    coeff_poc_fast = k.poc_fast * c.so4 * inhibition
    add_implicit_sink(
        LHS,
        RATES,
        "poc_slow",
        coeff_poc_slow,
        coeff_poc_slow * c.poc_slow,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )
    add_implicit_sink(
        LHS,
        RATES,
        "poc_fast",
        coeff_poc_fast,
        coeff_poc_fast * c.poc_fast,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # isotopes
    if mp.isotopes:
        alpha = 1.0 + (mp.msr_alpha - 1.0) * lim["so4_alpha_explicit"]
        coeff_so4_32 = calculate_fractionated_coeff_32(
            coeff_so4, c.so4, c.so4_32, alpha, eps=1e-30
        )

        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="so4_32",
            reactants={},
            products={"ts2_32": 1.0},
            coeff_master=coeff_so4_32,
            rate_master=coeff_so4_32 * c.so4_32,
            has_solid=has_solid,
            reaction_name="sulfate_reduction_32",
        )



def hs_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Reaction: 1 HS + 0.5 O2 -> 1 S0."""
    has_solid = False  # True if the reactants contain a solid phase species
    # H2S Sink - LIQUID
    # Ref: H2S
    coeff_ts2 = k.hs_ox * c.o2 * mp.hs_frac

    # O2 Sink (0.5x) - LIQUID
    coeff_o2 = 0.5 * k.hs_ox * c.ts2 * mp.hs_frac
    add_implicit_sink(
        LHS,
        RATES,
        "o2",
        coeff_o2,
        coeff_o2 * c.o2,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # S0 Source (1.0x) - SOLID, Couple to H2S
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="ts2",
        reactants={},
        products={"s0": 1.0},
        coeff_master=coeff_ts2,
        rate_master=coeff_ts2 * c.ts2,
        has_solid=has_solid,
        reaction_name="hs_oxidation",
    )

    if mp.isotopes:
        alpha = 1.0 + (mp.hs_ox_alpha - 1.0) * lim["ts2_alpha_explicit"]
        hs_32 = partition_equilibrium_isotope_32(
            c.ts2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        coeff_ts2_32 = calculate_fractionated_coeff_32(
            coeff_ts2, c.ts2 * mp.hs_frac, hs_32, alpha, eps=1e-20
        )

        # S0_32 coupled to H2S_32
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="ts2_32",
            reactants={},
            products={"s0_32": 1.0},
            coeff_master=coeff_ts2_32,
            rate_master=coeff_ts2_32 * c.ts2_32,
            has_solid=has_solid,
            reaction_name="hs_oxidation_32",
        )


def hs_oxidation_velde(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Reaction: 1 HS + 2 * O2 -> 1 SO4"""
    has_solid = False  # True if the reactants contain a solid phase species
    # H2S Sink - LIQUID
    # Ref: H2S
    coeff_ts2 = k.hs_ox * c.o2 * mp.hs_frac

    # O2 Sink (0.5x) - LIQUID
    coeff_o2 = 2 * k.hs_ox * c.ts2 * mp.hs_frac
    add_implicit_sink(
        LHS,
        RATES,
        "o2",
        coeff_o2,
        coeff_o2 * c.o2,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # S0 Source (1.0x) - SOLID, Couple to H2S
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="ts2",
        reactants={},
        products={"s0": 1.0},
        coeff_master=coeff_ts2,
        rate_master=coeff_ts2 * c.ts2,
        has_solid=has_solid,
        reaction_name="hs_oxidation_velde",
    )

    if mp.isotopes:
        alpha = 1.0 + (mp.hs_ox_alpha - 1.0) * lim["ts2_alpha_explicit"]
        hs_32 = partition_equilibrium_isotope_32(
            c.ts2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        coeff_ts2_32 = calculate_fractionated_coeff_32(
            coeff_ts2, c.ts2 * mp.hs_frac, hs_32, alpha, eps=1e-20
        )

        # S0_32 coupled to H2S_32
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="ts2_32",
            reactants={},
            products={"s0_32": 1.0},
            coeff_master=coeff_ts2_32,
            rate_master=coeff_ts2_32 * c.ts2_32,
            has_solid=has_solid,
            reaction_name="hs_oxidation_velde_32",
        )


def elemental_sulfur_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Reaction: 1 S0 + 1.5 O2 -> 1 SO4.

    Assuming that some O comes from H2O
    Phases: S0 (Solid), O2 (Liquid), SO4 (Liquid)
    """
    has_solid = True  # True if the reactants contain a solid phase species
    # S0 sink (Solid)
    # Rate = k * [O2] * [S0]
    coeff_s0 = k.s0_ox * c.o2

    # O2 Sink (1.5x) - LIQUID
    coeff_o2 = 1.5 * k.s0_ox * c.s0
    add_implicit_sink(
        LHS,
        RATES,
        "o2",
        coeff_o2,
        coeff_o2 * c.o2,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # SO4 Source (1.0x) - LIQUID, Coupled to S0 (SOLID)
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="s0",
        reactants={},
        products={"so4": 1.0},
        coeff_master=coeff_s0,
        rate_master=coeff_s0 * c.s0,
        has_solid=has_solid,
        reaction_name="elemental_sulfur_oxidation",
    )

    if mp.isotopes:
        # S0_32 Source (1.0x) - LIQUID, Coupled to S0_32 (SOLID)
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="s0_32",
            reactants={},
            products={"so4_32": 1.0},
            coeff_master=coeff_s0,
            rate_master=coeff_s0 * c.s0_32,
            has_solid=has_solid,
            reaction_name="elemental_sulfur_oxidation_32",
        )



        

def sulfide_mediated_iron_reduction_old(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Fe3 iron reduction via HS-.

    0.5 HS- + Fe3+ -> 0.5 S0 + Fe2+

    The reaction is doubly-capped so that at most 70% of either fe3 or ts2
    is consumed in a single timestep, guaranteeing perfect stoichiometry and
    unconditional positivity. Coupling uses a single master implicit variable
    (fe3) to ensure exact mass balance across all species.
    """
    has_solid = True  # True if the reactants contain a solid phase species
    # ------------------------------------------------------------------
    # 1. Current state (using FiPy variables directly for consistency)
    # ------------------------------------------------------------------
    ts2_val = nx.maximum(c.ts2, 0.0)
    hs_val = ts2_val * mp.hs_frac
    fe3_val = nx.maximum(c.fe3, 0.0)

    # ------------------------------------------------------------------
    # 2. Rate Calculation and Multi-Reactant Capping
    # ------------------------------------------------------------------
    k_base_val = k.fe3_hs * lim["o2_inhibit"] * lim["fe3_implicit"]

    # Uncapped reaction rate driven by fe3 consumption [mmol/L_solid/s]
    rate_uncapped = k_base_val * hs_val * fe3_val

    # Total available ts2 reservoir over the next timestep (including chemical production)
    # RATES contains rates in bulk units (mmol/L_bulk/s), so we divide by porosity to get mmol/L_liquid/s
    ts2_prod_pw = RATES["ts2"] / (mp.phi + 1e-30)
    dt_val = getattr(mp, "current_dt", 0.0)
    dt_safe = nx.maximum(dt_val, 1e-12)
    ts2_available = nx.maximum(ts2_val + ts2_prod_pw * dt_safe, 0.0)

    # Limit by fe3 depletion (at most 70% per timestep)
    max_rate_fe3 = 0.7 * fe3_val / dt_safe

    # Limit by ts2 depletion (0.5 mole ts2 consumed per 1 mole fe3)
    # 0.5 * Rate * dt <= 0.7 * ts2_available -> Rate <= 1.4 * ts2_available / dt
    max_rate_ts2 = 1.4 * ts2_available / dt_safe

    # Actual capped rate
    rate_actual = nx.minimum(rate_uncapped, nx.minimum(max_rate_fe3, max_rate_ts2))
    # rate_actual = rate_uncapped

    # Single master coefficient based on fe3 [1/s]
    coeff_master = rate_actual / (fe3_val + 1e-30)

    # ------------------------------------------------------------------
    # 3. Fe3 sink (Master Variable) — EXACTLY 1:1
    # ------------------------------------------------------------------
    add_implicit_sink(
        LHS,
        RATES,
        "fe3",
        coeff_master,
        rate_actual,
        mp=mp,
        has_solid=has_solid,
        c=c,
        reaction="sulfide_mediated_iron_reduction",
    )

    # ------------------------------------------------------------------
    # 4. Fe2 source (Coupled to fe3_new) — EXACTLY 1:1
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="fe2_total",
        source_species="fe3",
        coeff=coeff_master,
        rate=rate_actual,
        mp=mp,
        has_solid=has_solid,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=1.0,
        reaction="sulfide_mediated_iron_reduction",
    )

    # ------------------------------------------------------------------
    # 5. ts2 sink (Coupled to fe3_new) — EXACTLY 0.5:1
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="ts2",
        source_species="fe3",
        coeff=coeff_master,
        rate=rate_actual,
        mp=mp,
        has_solid=has_solid,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=-0.5,
        reaction="sulfide_mediated_iron_reduction",
    )

    # ------------------------------------------------------------------
    # 6. s0 source (Coupled to fe3_new) — EXACTLY 0.5:1
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="s0",
        source_species="fe3",
        coeff=coeff_master * 0.5,
        rate=0.5 * rate_actual,
        mp=mp,
        has_solid=has_solid,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=1.0,
        reaction="sulfide_mediated_iron_reduction",
    )

    # ------------------------------------------------------------------
    # 7. Isotopes (32S)
    # ------------------------------------------------------------------
    if mp.isotopes:
        hs_32 = partition_equilibrium_isotope_32(
            c.ts2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        f32 = hs_32 / (ts2_val * mp.hs_frac + 1e-30)

        rate_32 = 0.5 * rate_actual * f32
        coeff_32 = 0.5 * coeff_master * f32

        # ts2_32 sink
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="ts2_32",
            source_species="fe3",
            coeff=coeff_32,
            rate=rate_32,
            mp=mp,
            has_solid=has_solid,
            c=c,
            add_lhs_sink=False,
            stoich_ratio=-1.0,
        )

        # s0_32 source
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="s0_32",
            source_species="fe3",
            coeff=coeff_32,
            rate=rate_32,
            mp=mp,
            has_solid=has_solid,
            c=c,
            add_lhs_sink=False,
            stoich_ratio=1.0,
        )

        

def sulfide_mediated_iron_reduction_velde(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Fe3 iron reduction via HS-.

    Previsouly: 0.5 HS- + Fe3+ -> 0.5 S0 + Fe2+
    Velde: 1 HS- + 8 Fe3+ -> SO4 + 8 Fe2+

    The reaction is doubly-capped so that at most 70% of either fe3 or ts2
    is consumed in a single timestep, guaranteeing perfect stoichiometry and
    unconditional positivity. Coupling uses a single master implicit variable
    (fe3) to ensure exact mass balance across all species.
    """
    has_solid = True  # True if the reactants contain a solid phase species
    # ------------------------------------------------------------------
    # 1. Current state (using FiPy variables directly for consistency)
    # ------------------------------------------------------------------
    ts2_val = nx.maximum(c.ts2, 0.0)
    hs_val = ts2_val * mp.hs_frac
    fe3_val = nx.maximum(c.fe3, 0.0)

    # ------------------------------------------------------------------
    # 2. Rate Calculation and Multi-Reactant Capping
    # ------------------------------------------------------------------
    k_base_val = k.fe3_hs * lim["o2_inhibit"] * lim["fe3_implicit"]

    # Uncapped reaction rate driven by fe3 consumption [mmol/L_solid/s]
    rate_uncapped = k_base_val * hs_val * fe3_val

    # Total available ts2 reservoir over the next timestep (including chemical production)
    # RATES contains rates in bulk units (mmol/L_bulk/s), so we divide by porosity to get mmol/L_liquid/s
    ts2_prod_pw = RATES["ts2"] / (mp.phi + 1e-30)
    dt_val = getattr(mp, "current_dt", 0.0)
    dt_safe = nx.maximum(dt_val, 1e-12)
    ts2_available = nx.maximum(ts2_val + ts2_prod_pw * dt_safe, 0.0)

    # Limit by fe3 depletion (at most 70% per timestep)
    max_rate_fe3 = 0.7 * fe3_val / dt_safe

    # Limit by ts2 depletion (0.5 mole ts2 consumed per 1 mole fe3)
    # 0.5 * Rate * dt <= 0.7 * ts2_available -> Rate <= 1.4 * ts2_available / dt
    max_rate_ts2 = 1.4 * ts2_available / dt_safe

    # Actual capped rate
    rate_actual = nx.minimum(rate_uncapped, nx.minimum(max_rate_fe3, max_rate_ts2))
    # rate_actual = rate_uncapped

    # Single master coefficient based on fe3 [1/s]
    coeff_master = rate_actual / (fe3_val + 1e-30)

    # ------------------------------------------------------------------
    # 3. Fe3 sink (Master Variable) — EXACTLY 1:1
    # ------------------------------------------------------------------
    add_implicit_sink(
        LHS,
        RATES,
        "fe3",
        coeff_master,
        rate_actual,
        mp=mp,
        has_solid=has_solid,
        c=c,
        reaction="sulfide_mediated_iron_reduction",
    )

    # ------------------------------------------------------------------
    # 4. Fe2 source (Coupled to fe3_new) — EXACTLY 1:1
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="fe2_total",
        source_species="fe3",
        coeff=coeff_master,
        rate=rate_actual,
        mp=mp,
        has_solid=has_solid,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=1.0,
        reaction="sulfide_mediated_iron_reduction",
    )

    # ------------------------------------------------------------------
    # 5. so4 sink (Coupled to fe3_new) — EXACTLY 0.5:1
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="so4",
        source_species="fe3",
        coeff=coeff_master,
        rate=rate_actual,
        mp=mp,
        has_solid=has_solid,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=-1/8,
        reaction="sulfide_mediated_iron_reduction",
    )

    # ------------------------------------------------------------------
    # 7. Isotopes (32S)
    # ------------------------------------------------------------------
    if mp.isotopes:
        hs_32 = partition_equilibrium_isotope_32(
            c.ts2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        f32 = hs_32 / (ts2_val * mp.hs_frac + 1e-30)

        rate_32 = 1/8 * rate_actual * f32
        coeff_32 = 1/8 * coeff_master * f32

        # ts2_32 sink
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="ts2_32",
            source_species="fe3",
            coeff=coeff_32,
            rate=rate_32,
            mp=mp,
            has_solid=has_solid,
            c=c,
            add_lhs_sink=False,
            stoich_ratio=-1.0,
        )

        # so4 sink
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="so4_32",
            source_species="fe3",
            coeff=coeff_32,
            rate=rate_32,
            mp=mp,
            has_solid=has_solid,
            c=c,
            add_lhs_sink=False,
            stoich_ratio=1.0,
        )


def sulfide_mediated_iron_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    0.5 HS- + Fe3+ -> 0.5 S0 + Fe2+

    ts2 is the single master implicit variable.
    fe3, fe2, and s0 all cross-coupled to ts2_new:
      - exact stoichiometry between all species at any dt
      - fe3_implicit limiter in k_eff preserves diagonal dominance
    """
    import numpy as np

    has_solid = True  # True if the reactants contain a solid phase species
    # #  Suppress iron reduction in regions where FeS precipitation takes
    # #  precedence
    # #  Calculate FeS supersaturation
    # fe2_pw = c.fe2_total.value * mp.fe2_diss
    # hs_val = c.ts2.value * mp.hs_frac
    # omega = (fe2_pw * hs_val) / (k.hplus * k.fes_sp + 1e-30)
    # # Create a penalty factor (0 to 1)
    # # If omega > 1 (supersaturated), penalty drops quickly towards 0.
    # # If omega <= 1, penalty is 1 (normal iron reduction).
    # # fes_inhibit = 1.0 / nx.maximum(omega, 1.0)
    # # OR a steeper version:
    # fes_inhibit = nx.exp(-nx.maximum(omega - 1.0, 0.0))
    fes_inhibit = 1

    fe3_val = c.fe3.value
    ts2_val = c.ts2.value

    # ------------------------------------------------------------------
    # 1. Rate coefficient in L_pw basis (ts2 is liquid master)
    #    k_eff already contains hs_frac and fe3_implicit limiter
    # ------------------------------------------------------------------
    k_eff = (
        k.fe3_hs * mp.hs_frac * lim["o2_inhibit"] * lim["fe3_implicit"] * fes_inhibit
    )  # suppresses coeff as fe3 → 0

    coeff_ts2 = 0.5 * k_eff * fe3_val  # [1/s, L_pw] — ts2 master coeff

    # 2. Bulk Reaction coupling (using the wrapper)
    # Reaction stoichiometry normalized to 1 unit of TS2 consumed:
    # consumes: 1.0 ts2, 2.0 fe3
    # produces: 1.0 s0, 2.0 fe2_total
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="ts2",
        reactants={"fe3": 2.0},
        products={"s0": 1.0, "fe2_total": 2.0},
        coeff_master=coeff_ts2,
        rate_master=coeff_ts2 * ts2_val,
        has_solid=has_solid,
        reaction_name="sulfide_mediated_iron_reduction",
    )

    # 3. Isotope Reaction coupling (using the wrapper)
    if mp.isotopes:
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="ts2_32",
            reactants={},
            products={"s0_32": 1.0},
            coeff_master=coeff_ts2,
            rate_master=coeff_ts2 * c.ts2_32,
            has_solid=has_solid,
            reaction_name="sulfide_mediated_iron_reduction_32",
        )


def sulfide_speciation_clip(c, k, mp, dt, RATES, f=None):
    """Update reporting species (h2s, hs) based on total sulfide (ts2) and pH.

    This is FYI only and does not affect the reaction rates, which are all based on ts2.
    """
    c.h2s.setValue(c.ts2 * mp.h2s_frac)
    c.hs.setValue(c.ts2 * mp.hs_frac)

    if mp.isotopes:
        hs_frac_val = getattr(mp.hs_frac, "value", mp.hs_frac)
        h2s_frac_val = getattr(mp.h2s_frac, "value", mp.h2s_frac)
        alpha_val = getattr(mp.h2s_hs_alpha, "value", mp.h2s_hs_alpha)

        denom = hs_frac_val + alpha_val * h2s_frac_val + 1e-30
        c.hs_32.setValue(c.ts2_32 * hs_frac_val / denom)
        c.h2s_32.setValue(c.ts2_32 * alpha_val * h2s_frac_val / denom)


def fe2_sorption_clip(c, k, mp, dt, RATES, f=None):
    """Handle Iron Partitioning algebraically.

    Instead of calculating rates, we calculate fractions.

    System State: 'fe_total' is the primary variable.
    fe2 (liquid) and fe2_p (solid) are derived helper views.
    These are not used in the equations, and FYI only.
    """
    c.fe2.setValue(c.fe2_total * mp.fe2_diss)
    c.fe2_p.setValue(c.fe2_total * mp.fe2_sorb)


def fe2_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Reaction: 4 Fe2+ O2 -> 4 Fe3OOH.

    Note: Fe2_total tracks Fe2 liquid and sorbed. However the
    reaction rates are the same, so we use fe2_total
    """
    has_solid = False  # True if the reactants contain a solid phase species
    rate_base = k.fe2_ox * c.fe2_total * c.o2

    # O2 Sink (1/4) - LIQUID
    coeff_o2 = k.fe2_ox * c.fe2_total * 0.25
    add_implicit_sink(
        LHS,
        RATES,
        "o2",
        coeff_o2,
        rate_base * 0.25,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # Fe3 Source (1.0x) - SOLID, coupled to Fe2_total (liquid)
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        "fe3",  # product (solid)
        "fe2_total",  # source (liquid)
        k.fe2_ox * c.o2,  # coefficient
        rate_base,  # rate for reporting
        mp=mp,
        has_solid=has_solid,
        c=c,
    )


def fes_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 2.25 O2 -> 1 Fe3 + 1 SO4."""
    has_solid = True  # True if the reactants contain a solid phase species
    # FeS Sink - SOLID
    coeff_fes = k.fes_ox * c.o2

    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        "fe3",
        "fes",
        coeff_fes,
        coeff_fes * c.fes,
        mp=mp,
        has_solid=has_solid,
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
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # SO4 Source (1.0x) - LIQUID, Coupled to FeS (SOLID)
    # Use add_implicit_coupling_new with add_lhs_sink=False because the FeS sink
    # was already registered by the add_implicit_coupling_new call for fe3 above.

    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        "so4",  # target: liquid
        "fes",  # source: solid
        k.fes_ox * c.o2,
        coeff_fes * c.fes,
        mp=mp,
        has_solid=has_solid,
        c=c,
        add_lhs_sink=False,  # fes sink already added by the fe3 coupling above
    )

    if mp.isotopes:
        rate_base_32 = k.fes_ox * c.fes_32 * c.o2
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            "so4_32",
            "fes_32",
            k.fes_ox * c.o2,
            rate_base_32,
            mp=mp,
            has_solid=has_solid,
            c=c,
        )


def pyrite_formation_s0(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Reaction: 1 FeS + 1 S0 -> 1 FeS2.

    This is a bit tricky as we have two different S atoms into the same
    target (FeS2)
    """
    has_solid = True  # True if the reactants contain a solid phase species
    # S0 Sink - SOLID
    coeff_s0 = k.fes_s0 * c.fes
    add_implicit_sink(
        LHS,
        RATES,
        "s0",
        coeff_s0,
        coeff_s0 * c.s0,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # FeS to FeS2 SOLID, Rate = k * FeS * S0.
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="fes",
        reactants={},
        products={"fes2": 1.0},
        coeff_master=coeff_fes,
        rate_master=coeff_fes * c.fes,
        has_solid=has_solid,
        reaction_name="pyrite_formation_s0_fes",
    )

    if mp.isotopes:
        # S0 is porewater → must include mp.fac_s to match bulk sink coefficient,
        # and use "liquid_2_solid" for correct volume conversion to fes2_32 (solid)

        # 1st S atom: from s0_32 (porewater) to fes2_32 (solid)
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="s0_32",
            reactants={},
            products={"fes2_32": 1.0},
            coeff_master=k.fes_s0 * c.fes,
            rate_master=k.fes_s0 * c.fes * c.s0_32,
            has_solid=has_solid,
            reaction_name="pyrite_formation_s0_32",
        )

        # 2nd S atom: from fes_32 (solid) to fes2_32 (solid)
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="fes_32",
            reactants={},
            products={"fes2_32": 1.0},
            coeff_master=k.fes_s0 * c.s0,
            rate_master=k.fes_s0 * c.s0 * c.fes_32,
            has_solid=has_solid,
            reaction_name="pyrite_formation_fes_32",
        )


def pyrite_formation_fes_ts2(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Reaction: 1 FeS + 1 HS -> 1 FeS2Fe."""
    has_solid = True  # True if the reactants contain a solid phase species
    # H2S Sink (1.0x) - LIQUID
    coeff_ts2 = k.fes_ts2 * c.fes * mp.hs_frac
    add_implicit_sink(
        LHS,
        RATES,
        "ts2",
        coeff_ts2,
        coeff_ts2 * c.ts2,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # FeS2 Source (1.0x) - SOLID
    # Couple to FeS.
    # Rate = k * H2S * FeS.
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="fes",
        reactants={},
        products={"fes2": 1.0},
        coeff_master=k.fes_ts2 * c.ts2 * mp.hs_frac,
        rate_master=k.fes_ts2 * c.ts2 * c.fes * mp.hs_frac,
        has_solid=has_solid,
        reaction_name="pyrite_formation_fes_ts2",
    )

    if mp.isotopes:
        # 1st S atom: from FeS_32 (solid) -> FeS2_32 (solid)
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="fes_32",
            reactants={},
            products={"fes2_32": 1.0},
            coeff_master=k.fes_ts2 * c.ts2 * mp.hs_frac,
            rate_master=k.fes_ts2 * c.ts2 * c.fes_32 * mp.hs_frac,
            has_solid=has_solid,
            reaction_name="pyrite_formation_fes_ts2_32_solid",
        )

        # 2nd S atom: from H2S_32 (ts2_32) (liquid) -> FeS2_32 (solid)
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="ts2_32",
            reactants={},
            products={"fes2_32": 1.0},
            coeff_master=coeff_ts2,
            rate_master=coeff_ts2 * c.ts2_32,
            has_solid=has_solid,
            reaction_name="pyrite_formation_fes_ts2_32_liquid",
        )


def pyrite_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Reaction: 1 FeS2 + 3.5 O2 → 1 Fe3 + 2 SO4.

    Coupling strategy:
      - SO4 is cross-coupled to O2  (liquid_2_liquid, stoich 2/3.5)
      - Fe3 is cross-coupled to fes2 (solid_2_solid, stoich 1.0)
    """
    has_solid = True  # True if the reactants contain a solid phase species
    # ------------------------------------------------------------------
    # 1. Base coefficients
    # ------------------------------------------------------------------
    # O2 is porewater master: coeff_o2 is implicit sink on o2
    coeff_o2 = 3.5 * k.fes2_ox * c.fes2  # [L_pw basis]
    coeff_fes2 = k.fes2_ox * c.o2
    rate_o2 = coeff_o2 * c.o2  # mol/L_pw/s
    rate_fes2 = coeff_fes2 * c.fes2  # mol/L_pw/s

    # 1. O2 -> SO4 (2/3.5 ratio)
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="o2",
        reactants={},
        products={"so4": 2.0 / 3.5},
        coeff_master=coeff_o2,
        rate_master=rate_o2,
        has_solid=has_solid,
        reaction_name="pyrite_oxidation_o2",
    )

    # 2. FeS2 -> Fe3
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="fes2",
        reactants={},
        products={"fe3": 1.0},
        coeff_master=coeff_fes2,
        rate_master=rate_fes2,
        has_solid=has_solid,
        reaction_name="pyrite_oxidation_fes2",
    )

    # 3. Isotopes: FeS2_32 -> SO4_32
    if mp.isotopes:
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="fes2_32",
            reactants={},
            products={"so4_32": 1.0},
            coeff_master=coeff_fes2,
            rate_master=coeff_fes2 * c.fes2_32,
            has_solid=has_solid,
            reaction_name="pyrite_oxidation_32",
        )


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
    has_solid = False  # True if the reactants contain a solid phase species
    # ------------------------------------------------------------------
    # 1. Current state
    # ------------------------------------------------------------------
    fe2_pw_val = c.fe2_total * mp.fe2_diss + 1e-20
    ts2_val = c.ts2
    hs_val = ts2_val * mp.hs_frac

    # ------------------------------------------------------------------
    # 2. Quadratic bound
    #
    #   (fe2 - X)(ts2 - X) = K_eff
    #   X_max = ½[(fe2+ts2) - √((fe2-ts2)² + 4·K_eff)]
    # ------------------------------------------------------------------
    omega_den = k.hplus * k.fes_sp + 1e-30
    K_eff = omega_den / (mp.hs_frac + 1e-30)
    discriminant = (fe2_pw_val - ts2_val) ** 2 + 4.0 * K_eff
    X_max = 0.5 * ((fe2_pw_val + ts2_val) - nx.sqrt(discriminant))
    X_max = nx.maximum(X_max, 0.0)

    ts2_target = ts2_val - X_max

    # ------------------------------------------------------------------
    # 3. Regime flag and rate coefficient
    #
    #   k_rxn includes hs_frac (v9 convention): rate scales with HS
    #   availability.  No fe2_limiter needed — X_max bounds the target.
    # ------------------------------------------------------------------
    omega = (fe2_pw_val * hs_val) / omega_den
    sharpness = 100.0
    is_precip = 0.5 * (1.0 + nx.tanh(sharpness * (omega - 1.0)))

    k_rxn = k.fes_isp * is_precip * mp.hs_frac  # [1/s, L_pw]

    # ------------------------------------------------------------------
    # 4. Dynamic Depletion Cap (Safety Net to prevent negative Fe2/TS2)
    # ------------------------------------------------------------------
    dt_val = getattr(mp, "current_dt", 0.0)
    dt_safe = nx.maximum(dt_val, 1e-12)

    fe2_total_val = nx.maximum(c.fe2_total, 0.0)
    fe2_prod_pw = RATES["fe2_total"] / (mp.phi + 1e-30)
    ts2_prod_pw = RATES["ts2"] / (mp.phi + 1e-30)

    fe2_available = nx.maximum(fe2_total_val + fe2_prod_pw * dt_val, 0.0)
    ts2_available = nx.maximum(ts2_val + ts2_prod_pw * dt_val, 0.0)

    # We allow the reactant to deplete up to 99.9999% of its available pool,
    # ensuring we can get close to true equilibrium (omega = 1) without going negative.
    rate_cap_fe2 = 0.999999 * fe2_available / dt_safe
    rate_cap_ts2 = 0.999999 * ts2_available / dt_safe

    # Potential rate to reach equilibrium target ts2_target (since change is X_max)
    rate_pot = k_rxn * X_max

    # Capped rate
    precip_rate = nx.minimum(rate_pot, nx.minimum(rate_cap_fe2, rate_cap_ts2))
    precip_rate = nx.maximum(precip_rate, 0.0)

    # Effective rate coefficient (k_rxn_eff) to use in implicit sink and coupling terms.
    # If X_max is zero, we fall back to k_rxn (which is also zero).
    k_rxn_eff = nx.where(X_max > 1e-30, precip_rate / (X_max + 1e-30), k_rxn)

    # ------------------------------------------------------------------
    # 5a. ts2 — self-implicit relaxation toward ts2_target
    #
    #   d[ts2]/dt = -k_rxn_eff·ts2 + k_rxn_eff·ts2_target
    #   ts2_new → weighted avg of ts2_old and ts2_target → bounded
    # ------------------------------------------------------------------
    add_implicit_sink(
        LHS, RATES, "ts2", k_rxn_eff, precip_rate, mp=mp, has_solid=has_solid
    )
    add_explicit_source(
        RHS,
        RATES,
        "ts2",
        k_rxn_eff * ts2_target,
        update_rates=False,
        mp=mp,
        has_solid=has_solid,
        reaction="fes_precipitation",
    )

    # ------------------------------------------------------------------
    # 5b. fe2_total — CROSS to ts2_new (v9 convention)
    #
    #   Same coefficient as ts2 sink → 1:1 stoichiometry
    #   Δfe2 = Δts2 (both deplete by same amount)
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="fe2_total",
        source_species="ts2",
        coeff=k_rxn_eff,
        rate=precip_rate,
        mp=mp,
        has_solid=has_solid,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=-1.0,
    )

    add_explicit_source(
        RHS,
        RATES,
        "fe2_total",
        k_rxn_eff * ts2_target,
        update_rates=False,
        mp=mp,
        has_solid=has_solid,
    )

    # ------------------------------------------------------------------
    # 5c. fes — CROSS to ts2_new (v9 convention)
    #
    #   Same coefficient, opposite sign.  NO fac_s!
    #   eff_phi on transient term handles phase-volume conversion.
    # ------------------------------------------------------------------
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="fes",
        source_species="ts2",
        coeff=k_rxn_eff,
        rate=precip_rate,
        mp=mp,
        has_solid=has_solid,
        c=c,
        add_lhs_sink=False,
        stoich_ratio=1.0,
    )
    add_explicit_source(
        RHS,
        RATES,
        "fes",
        -k_rxn_eff * ts2_target,
        update_rates=False,
        mp=mp,
        has_solid=has_solid,
    )

    # ------------------------------------------------------------------
    # 6. Isotopes (32S)
    # ------------------------------------------------------------------
    if mp.isotopes:
        hs_32 = partition_equilibrium_isotope_32(
            c.ts2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        f32_hs = hs_32 / (ts2_val * mp.hs_frac + 1e-30)

        ts2_32_target = ts2_target * f32_hs
        precip_rate_32 = k_rxn_eff * nx.maximum(c.ts2_32 - ts2_32_target, 0.0)

        # ts2_32: self-implicit
        add_implicit_sink(
            LHS, RATES, "ts2_32", k_rxn_eff, precip_rate_32, mp=mp, has_solid=has_solid
        )
        add_explicit_source(
            RHS,
            RATES,
            "ts2_32",
            k_rxn_eff * ts2_32_target,
            update_rates=False,
            mp=mp,
            has_solid=has_solid,
        )

        # fes_32: CROSS to ts2_32
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="fes_32",
            source_species="ts2_32",
            coeff=k_rxn_eff,
            rate=precip_rate_32,
            mp=mp,
            has_solid=has_solid,
            c=c,
            add_lhs_sink=False,
        )
        add_explicit_source(
            RHS,
            RATES,
            "fes_32",
            -k_rxn_eff * ts2_32_target,
            mp=mp,
            update_rates=False,
            has_solid=has_solid,
        )


def fes_precipitation_terminal(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """FeS precipitation as terminal solid phase. No dissolution.

    Fe2 + HS- -> FeS

    All bulk species and fes_32 driven by fe2_new.
    ts2_32 has its own self-implicit sink with the same coefficient
    as bulk ts2, preserving the isotope ratio during depletion.
    """
    has_solid = False
    # 1. Current state (using FiPy variables directly for consistency)
    fe2_total_val = nx.maximum(c.fe2_total, 0.0)
    ts2_val = nx.maximum(c.ts2, 0.0)

    fe2_pw_val = fe2_total_val * mp.fe2_diss + 1e-20
    hs_val = ts2_val * mp.hs_frac
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_pw_val * hs_val) / omega_den

    # 2. Define the 'Driving Force' (only positive for precipitation)
    driving_force = nx.maximum(omega - 1.0, 0.0)

    # 3. Apply the MM-type limiter
    km_fes = 0.5
    equilibrium_limiter = driving_force / (km_fes + driving_force)

    # Velde's phase-specific rate [mol/m^3_pw/s]
    rate_pw_uncapped = k.fes_isp * equilibrium_limiter

    # H2S and Fe2+ net chemical production rates from other reactions (in porewater phase-specific units)
    fe2_prod_pw = RATES["fe2_total"] / (mp.phi + 1e-30)
    ts2_prod_pw = RATES["ts2"] / (mp.phi + 1e-30)

    # Total available reservoir over the next timestep (including chemical production)
    dt_val = getattr(mp, "current_dt", 0.0)
    dt_safe = nx.maximum(dt_val, 1e-12)
    fe2_available = nx.maximum(fe2_total_val + fe2_prod_pw * dt_val, 0.0)
    ts2_available = nx.maximum(ts2_val + ts2_prod_pw * dt_val, 0.0)

    # Timestep-dependent depletion caps (prevent depleting > 99% of total available species per step)
    rate_cap_fe2 = 0.99 * fe2_available / dt_safe
    rate_cap_ts2 = 0.99 * ts2_available / dt_safe

    rate_pw = nx.minimum(rate_pw_uncapped, nx.minimum(rate_cap_fe2, rate_cap_ts2))

    # Implicit coefficients based on solved variables
    fes_coeff = rate_pw / (fe2_total_val + 1e-5)
    hs_coeff = rate_pw / (ts2_val + 1e-5)

    # Fe2 sink + FeS source: CROSS to fe2_new
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="fes",
        source_species="fe2_total",
        coeff=fes_coeff,
        rate=rate_pw,
        mp=mp,
        has_solid=has_solid,
        c=c,
        add_lhs_sink=True,
        stoich_ratio=1.0,
        reaction="fes_precipitation",
    )

    # TS2 sink: self-implicit
    add_implicit_sink(
        LHS,
        RATES,
        "ts2",
        hs_coeff,
        rate_pw,
        mp=mp,
        has_solid=has_solid,
        reaction="fes_precipitation",
    )

    if mp.isotopes:
        hs_32 = partition_equilibrium_isotope_32(
            c.ts2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        f32_hs = hs_32 / (ts2_val * mp.hs_frac + 1e-30)
        rate_pw_32 = rate_pw * f32_hs

        # fes_32: CROSS to fe2_new
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="fes_32",
            source_species="fe2_total",
            coeff=fes_coeff * f32_hs,
            rate=rate_pw_32,
            mp=mp,
            has_solid=has_solid,
            c=c,
            add_lhs_sink=False,  # fe2 sink already registered above
        )

        # ts2_32: self-implicit sink, same coeff as bulk ts2
        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            hs_coeff,
            rate_pw_32,
            mp=mp,
            has_solid=has_solid,
        )


def fes_dissolution(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    FeS dissolution — equilibrium capped.

    All rates in natural model units (mmol/L_pw or mmol/L_solid).
    Porosity conversions are handled transparently via the has_solids key

    using michaelis menten limiter
    """
    has_solid = True  # True if the reactants contain a solid phase species
    # ------------------------------------------------------------------
    # 1. Current state
    # ------------------------------------------------------------------
    fe2_pw_val = c.fe2_total * mp.fe2_diss + 1e-20  # mmol/L_pw
    ts2_val = c.ts2  # mmol/L_pw
    hs_val = ts2_val * mp.hs_frac  # mmol/L_pw
    fes_val = c.fes  # mmol/L_solid

    # 1. Calculate Omega
    km_diss = 0.5
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = (fe2_pw_val * hs_val) / omega_den
    zero = c.fes * 0.0  # quick hack to get a fipy cellvariable
    undersat = nx.maximum(zero, 1.0 - omega)
    dissol_limiter = undersat / (km_diss + undersat)

    # ------------------------------------------------------------------
    # 3. Thermodynamic Capacity Limiter (The Over-shoot Fix)
    # ------------------------------------------------------------------
    # Calculate the exact porewater mass increment (dx_eq) required to hit Omega = 1
    # Quadratic: dx^2 + (fe2 + hs)*dx + (fe2*hs - Ksp) = 0
    # standard form: a=1, b=(fe2 + hs), c=(fe2*hs - K_sp_prime)

    k_sp_prime = omega_den  # This is k.hplus * k.fes_sp

    b_coef = fe2_pw_val + hs_val
    c_coef = (fe2_pw_val * hs_val) - k_sp_prime

    # Radical: sqrt(b^2 - 4ac)
    discriminant = b_coef**2 - 4.0 * c_coef

    # Safe guard against negative noise under radical
    discriminant = nx.maximum(zero, discriminant)

    # Solve quadratic for the positive root
    dx_eq = (-b_coef + nx.sqrt(discriminant)) / 2.0
    # If already supersaturated, dx_eq will be <= 0. Clamp it.
    dx_eq = nx.maximum(zero, dx_eq)

    # Convert dx_eq from [mmol/L_pw] back to solid phase equivalent [mmol/L_solid]
    # to compare against our available solid inventory.
    # Conversion: solid = liquid * (phi / (1 - phi))
    fac_l_to_s = mp.phi / (1.0 - mp.phi)
    dx_eq_solid = dx_eq * fac_l_to_s

    # ------------------------------------------------------------------
    # 4. Combine Inventory Cap and Thermodynamic Cap safely
    # ------------------------------------------------------------------
    # The maximum mass allowed to dissolve this timestep is the lesser of:
    # 1. A fraction of the solid inventory (f_max * fes_val)
    # 2. The thermodynamic space remaining in the porewater (dx_eq_solid)

    f_max = 0.5
    max_allowed_mass_solid = nx.minimum(f_max * fes_val, dx_eq_solid)

    # Define a safety timescale over which we want to smoothly land on equilibrium.
    # E.g., land smoothly over ~15 minutes (900 seconds) or current_dt, whichever is smaller.
    # tau_safe = nx.minimum(mp.current_dt, 900.0)
    # Define a relaxation timescale proportional to the timestep
    # Setting it to 5 * dt ensures the system under-relaxingly glides
    # toward equilibrium over multiple timesteps rather than violently forcing it.
    tau_safe = 5.0 * mp.current_dt

    # Back-calculate the maximum allowed safe k_d coefficient
    # rate = k_d * fes_val  ==>  k_d_max = max_allowed_mass_solid / (fes_val * tau_safe)
    k_d_max = max_allowed_mass_solid / (fes_val * tau_safe + 1e-30)

    # Apply the thermodynamic cap to your kinetic rate constant
    k_d = k.fes_isd * dissol_limiter
    k_d = nx.minimum(k_d, k_d_max)

    # Final Patankar-style step-limiter for standalone numerical matrix stability
    k_d = k_d / (1.0 + (k_d * mp.current_dt / f_max))

    # ------------------------------------------------------------------
    # 5. Dissolution rate [mmol/L_solid/s]
    # ------------------------------------------------------------------
    diss_rate_solid = k_d * fes_val

    # print(f"f max = {nx.max(diss_rate_solid.value):.2e}, f min = {nx.min(diss_rate_solid.value):.2e}")

    # 5. FeS dissolution coupling (using the wrapper)
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="fes",
        reactants={},
        products={"ts2": 1.0, "fe2_total": 1.0},
        coeff_master=k_d,
        rate_master=diss_rate_solid,
        has_solid=has_solid,
        reaction_name="fes_dissolution",
    )

    # 6. Isotopes (32S) — no fractionation, k_d identical to bulk
    if mp.isotopes:
        fes_32_val = c.fes_32
        diss_32_solid = k_d * fes_32_val

        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="fes_32",
            reactants={},
            products={"ts2_32": 1.0},
            coeff_master=k_d,
            rate_master=diss_32_solid,
            has_solid=has_solid,
            reaction_name="fes_dissolution_32",
        )


def s0_disproportionation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate elemental sulfur disproportionation.

    Reaction: 3S0 + 8H2O -> H2S + 2SO4

    Notes
    -----
    - The split between H2S and SO4 depends on mp.dispro_so4_hs_split,
      which is the ratio between H2S/SO4, typically 1:2 -> 0.5
    - The isotope fractionation between S0 and H2S is given by mp.dispro_hs_alpha (0.993)
    - The isotope fractionation between S0 and SO4 is given by mp.dispro_so4_alpha (1.02)
    - The reaction constant for the overall reaction is given by k.s0_dispro
    - The reaction rate depends on S0, and H2S & O2 as inhibitors.
    """
    has_solid = True  # True if the reactants contain a solid phase species
    # 1. Base Rate Calculation (Master Species: S0)
    # Disproportionation is anaerobic, so O2 is NOT a reactant.
    # Instead, we use the inhibitor lim["disp_o2_inhibit"] to ensure it only
    # proceeds under low oxygen conditions.
    rate_uncapped = k.s0_dispro * c.s0 * lim["ts2"] * lim["o2_inhibit"]

    # Capping to prevent over-consumption in a single timestep
    max_rate_s0 = 0.7 * c.s0 / (mp.current_dt + 1e-30)
    rate_actual = nx.minimum(rate_uncapped, max_rate_s0)

    # Coefficient based on the consumed species (S0)
    coeff_s0_base = rate_actual / (c.s0 + 1e-30)

    # 2. Calculate the Stoichiometric Split
    # If split = 0.5 (1 H2S : 2 SO4), then for 1.5 moles of S0:
    # 1.0 mole goes to SO4 and 0.5 moles go to H2S
    split = mp.dispro_so4_hs_split
    so4_fraction = 1.0 / (1.0 + split)
    h2s_fraction = split / (1.0 + split)

    # S0 Disproportionation coupling (using the wrapper)
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="s0",
        reactants={},
        products={"so4": so4_fraction, "ts2": h2s_fraction},
        coeff_master=coeff_s0_base,
        rate_master=coeff_s0_base * c.s0,
        has_solid=has_solid,
        reaction_name="s0_disproportionation",
    )

    # O2 Consumption - REMOVED (Disproportionation is anaerobic)
    # If the user intended this to be S0 oxidation, O2 should be a reactant.
    # But for disproportionation, it is purely internal redox.

    # 5. Isotopes
    if mp.isotopes:
        # To maintain isotope mass balance, the 32S leaving S0 must exactly equal
        # the 32S entering H2S and SO4. If the user-provided alphas do not have a
        # weighted average of 1.0, mass is created/destroyed. We normalize them here:
        weighted_alpha = (
            h2s_fraction * mp.dispro_hs_alpha + so4_fraction * mp.dispro_so4_alpha
        )
        norm_hs_alpha = mp.dispro_hs_alpha / weighted_alpha
        norm_so4_alpha = mp.dispro_so4_alpha / weighted_alpha

        # Fractionation for H2S path
        coeff_hs_32 = calculate_fractionated_coeff_32(
            coeff_ts2, c.s0, c.s0_32, norm_hs_alpha, eps=1e-30
        )

        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="s0_32",
            reactants={},
            products={"ts2_32": 1.0},
            coeff_master=coeff_hs_32,
            rate_master=coeff_hs_32 * c.s0_32,
            has_solid=has_solid,
            reaction_name="s0_disproportionation_32_hs",
        )

        # Fractionation for SO4 path
        coeff_so4_32 = calculate_fractionated_coeff_32(
            coeff_so4, c.s0, c.s0_32, norm_so4_alpha, eps=1e-30
        )

        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="s0_32",
            reactants={},
            products={"so4_32": 1.0},
            coeff_master=coeff_so4_32,
            rate_master=coeff_so4_32 * c.s0_32,
            has_solid=has_solid,
            reaction_name="s0_disproportionation_32_so4",
        )


def fes_precipitation_dissolution_linearized(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    FeS precipitation / dissolution with Newton-linearised saturation ratio Ω.

    Precipitation (Ω ≥ 1):  Fe2⁺ + HS⁻  →  FeS(s)
        R_prec = k_prec_eff · (Ω^{n+1} - 1)

    Dissolution   (Ω < 1):  FeS(s)  →  Fe2⁺ + HS⁻
        R_diss = k_diss · fes · (1 - Ω^{n+1})

    Ω = (fe2_total · fe2_diss · ts2 · hs_frac) / (H⁺ · Ksp)

    Newton linearisation around the current sweep iterate (*):
        Ω^{n+1} ≈ ω* + dO_dFe2·(fe2^{n+1} − fe2*)
                      + dO_dTS2·(ts2^{n+1} − ts2*)

    Each branch maps to three contributions:
        (a) implicit diagonal  → add_implicit_coupling_new / add_implicit_sink
        (b) implicit cross     → direct CROSS.append (helpers only wire one diagonal per call)
        (c) explicit residual  → add_explicit_source  (can be negative: corrects double-count)

    Key differences vs. original fes_precipitation_terminal:
        - No Michaelis-Menten limiter (stability from implicit scheme, not rate capping)
        - No dt-dependent rate capping (eliminated by design)
        - Dissolution branch added; requires k.fes_diss in the rate constant object
        - Requires sweep loop: call .sweep() 3-10× per timestep until residuals converge
    """
    phi = mp.phi
    has_solid = True  # True for dissolution branch (solid reactant FeS)
    has_solid_prec = (
        False  # False for precipitation branch (liquid reactants Fe2_pw, HS-)
    )

    # ── Current sweep iterate ─────────────────────────────────────────────
    fe2_val = nx.maximum(c.fe2_total.value, 1e-20)
    ts2_val = nx.maximum(c.ts2.value, 1e-20)
    fes_val = nx.maximum(c.fes.value, 1e-20)

    fe2_pw = fe2_val * mp.fe2_diss
    hs_val = ts2_val * mp.hs_frac
    omega_den = k.hplus * k.fes_sp + 1e-30
    omega = fe2_pw * hs_val / omega_den

    # ── ∂Ω/∂(transported variable) ───────────────────────────────────────
    # ∂Ω/∂(fe2_total) = fe2_diss · hs_val  / omega_den
    # ∂Ω/∂(ts2)       = fe2_pw   · hs_frac / omega_den
    # Useful identity: dO_dFe2·fe2_val = dO_dTS2·ts2_val = ω*  (symmetry check)
    dO_dFe2 = mp.fe2_diss * hs_val / omega_den
    dO_dTS2 = fe2_pw * mp.hs_frac / omega_den

    # Explicit residual after factoring out the two implicit terms:
    #   omega_res = ω* − dO_dFe2·fe2* − dO_dTS2·ts2* = ω* − ω* − ω* = −ω*
    # Kept as a general expression rather than the analytic shortcut for
    # numerical safety near zero concentrations.
    omega_res = omega - dO_dFe2 * fe2_val - dO_dTS2 * ts2_val

    # ── Per-cell branch selector (no dt dependence) ───────────────────────
    is_prec = (omega >= 1.0).astype(float)
    is_diss = 1.0 - is_prec

    # ══════════════════════════════════════════════════════════════════════
    # PRECIPITATION BRANCH
    #   R_prec = k_prec_eff · (Ω^{n+1} − 1)
    #          ≈ k_prec_eff · [dO_dFe2·fe2  +  dO_dTS2·ts2  +  (omega_res − 1)]
    #                          ──── (a) ────    ──── (b) ────    ──── (c) ────
    # ══════════════════════════════════════════════════════════════════════
    k_prec_eff = k.fes_isp * is_prec  # zero in dissolution cells (no fac_vol!)

    # ── (a) Implicit diagonal in fe2_total ───────────────────────────────
    prec_coeff_fe2 = k_prec_eff * dO_dFe2  # [1/s]
    prec_rate_fe2 = prec_coeff_fe2 * fe2_val

    add_implicit_coupling_new(  # fes  +=  prec_coeff_fe2 · fe2_total (CROSS)
        CROSS,
        RATES,
        LHS,  # fe2_total  −=  prec_coeff_fe2       (LHS diagonal)
        target_species="fes",
        source_species="fe2_total",
        coeff=prec_coeff_fe2,
        rate=prec_rate_fe2,
        mp=mp,
        has_solid=has_solid_prec,
        add_lhs_sink=True,
        stoich_ratio=1.0,
    )

    # ── (b) Implicit cross in ts2 ────────────────────────────────────────
    prec_coeff_ts2 = k_prec_eff * dO_dTS2
    prec_rate_ts2 = prec_coeff_ts2 * ts2_val

    add_implicit_coupling_new(  # fes  +=  prec_coeff_ts2 · ts2       (CROSS)
        CROSS,
        RATES,
        LHS,  # ts2 diagonal handled separately below
        target_species="fes",
        source_species="ts2",
        coeff=prec_coeff_ts2,
        rate=prec_rate_ts2,
        mp=mp,
        has_solid=has_solid_prec,
        add_lhs_sink=False,
        stoich_ratio=1.0,
    )
    add_implicit_sink(  # ts2  −=  prec_coeff_ts2 · ts2       (LHS diagonal)
        LHS,
        RATES,
        "ts2",
        prec_coeff_ts2,
        prec_rate_ts2,
        mp=mp,
        has_solid=has_solid_prec,
    )

    # ── Stoichiometric off-diagonal sinks ────────────────────────────────
    # add_implicit_coupling_new wires one diagonal per call.
    # The full Jacobian also requires:
    #   fe2_total consumed proportionally to ts2  →  A[fe2, ts2] block
    #   ts2       consumed proportionally to fe2  →  A[ts2, fe2] block
    # Sign: negative = off-diagonal consumption.
    CROSS["fe2_total"].append(("ts2", -prec_coeff_ts2 * phi))
    CROSS["ts2"].append(("fe2_total", -prec_coeff_fe2 * phi))

    # ── (c) Explicit residual ─────────────────────────────────────────────
    # prec_res_rate = k_prec_eff · (omega_res − 1) = k_prec_eff · (−ω* − 1)  < 0
    # Negative by construction: corrects for the ω* that was double-counted in
    # (a) and (b).  At convergence: (a) + (b) + (c) = R_prec exactly.
    prec_res_rate = k_prec_eff * (omega_res - 1.0)

    add_explicit_source(
        RHS, RATES, "fes", prec_res_rate, mp=mp, has_solid=has_solid_prec
    )
    add_explicit_source(
        RHS, RATES, "fe2_total", -prec_res_rate, mp=mp, has_solid=has_solid_prec
    )
    add_explicit_source(
        RHS, RATES, "ts2", -prec_res_rate, mp=mp, has_solid=has_solid_prec
    )

    # ══════════════════════════════════════════════════════════════════════
    # DISSOLUTION BRANCH
    #   R_diss = k_diss · fes · (1 − Ω^{n+1})
    #
    # Full Newton linearisation:
    #   R_diss ≈ k_diss · [(1−ω*)·fes^{n+1}              ← (a) fes self-implicit
    #                    − fes* · dO_dFe2·fe2^{n+1}      ← (b) dissolution suppressed by Fe2
    #                    − fes* · dO_dTS2·ts2^{n+1}      ← (c) dissolution suppressed by TS2
    #                    + fes* · (dO_dFe2·fe2* + dO_dTS2·ts2*)]  ← (d) explicit
    #
    # (a) is the critical term that breaks the circular FeS dependency.
    # (b,c,d) encode the Ω feedback: as Fe2/TS2 rise, Ω rises, dissolution slows.
    # ══════════════════════════════════════════════════════════════════════
    k_diss_eff = k.fes_isd * is_diss  # zero in precipitation cells

    # ── (a) FeS self-implicit: the circular-dependency fix ────────────────
    diss_coeff_fes = k_diss_eff * (1.0 - omega)  # positive since ω < 1 in diss cells
    diss_rate_fes = diss_coeff_fes * fes_val

    add_implicit_coupling_new(  # fe2_total  +=  diss_coeff_fes · fes  (CROSS)
        CROSS,
        RATES,
        LHS,  # fes        −=  diss_coeff_fes        (LHS diagonal)
        target_species="fe2_total",
        source_species="fes",
        coeff=diss_coeff_fes,
        rate=diss_rate_fes,
        mp=mp,
        has_solid=has_solid,
        add_lhs_sink=True,
        stoich_ratio=1.0,
    )
    add_implicit_coupling_new(  # ts2  +=  diss_coeff_fes · fes        (CROSS)
        CROSS,
        RATES,
        LHS,  # fes diagonal already registered above
        target_species="ts2",
        source_species="fes",
        coeff=diss_coeff_fes,
        rate=diss_rate_fes,
        mp=mp,
        has_solid=has_solid,
        add_lhs_sink=False,
        stoich_ratio=1.0,
    )

    # ── (b,c) Cross-suppression: dissolution rate decreases as Fe2/TS2 rise ─
    # diss_cross_* = k_diss · fes* · ∂Ω/∂variable  [1/s]
    diss_cross_fe2 = k_diss_eff * fes_val * dO_dFe2
    diss_cross_ts2 = k_diss_eff * fes_val * dO_dTS2

    # fes: positive cross = less net dissolution when Fe2/TS2 rise (source inhibition)
    CROSS["fes"].append(("fe2_total", +diss_cross_fe2 * (1.0 - phi)))
    CROSS["fes"].append(("ts2", +diss_cross_ts2 * (1.0 - phi)))
    # fe2, ts2: negative cross = less source production as Fe2/TS2 rise
    CROSS["fe2_total"].append(("fe2_total", -diss_cross_fe2 * phi))
    CROSS["fe2_total"].append(("ts2", -diss_cross_ts2 * phi))
    CROSS["ts2"].append(("fe2_total", -diss_cross_fe2 * phi))
    CROSS["ts2"].append(("ts2", -diss_cross_ts2 * phi))

    # ── (d) Explicit constant for dissolution ─────────────────────────────
    # = k_diss · fes* · (dO_dFe2·fe2* + dO_dTS2·ts2*)
    # = k_diss · fes* · 2ω*  (via the identity above)
    diss_res_rate = k_diss_eff * fes_val * (dO_dFe2 * fe2_val + dO_dTS2 * ts2_val)

    add_explicit_source(
        RHS, RATES, "fe2_total", diss_res_rate, mp=mp, has_solid=has_solid
    )
    add_explicit_source(RHS, RATES, "ts2", diss_res_rate, mp=mp, has_solid=has_solid)
    add_explicit_source(RHS, RATES, "fes", -diss_res_rate, mp=mp, has_solid=has_solid)

    # ── Isotopes ──────────────────────────────────────────────────────────
    if mp.isotopes:
        hs_32 = partition_equilibrium_isotope_32(
            c.ts2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        f32_hs = hs_32 / (hs_val + 1e-30)

        # fes_32 precipitation: coupled to fe2_total, same coefficient scaled by isotope fraction
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="fes_32",
            source_species="fe2_total",
            coeff=prec_coeff_fe2 * f32_hs,
            rate=prec_rate_fe2 * f32_hs,
            mp=mp,
            has_solid=has_solid_prec,
            add_lhs_sink=False,  # fe2_total diagonal already registered
            stoich_ratio=1.0,
        )
        # ts2_32 self-implicit sink: same diagonal coefficient as bulk ts2
        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            prec_coeff_ts2,
            prec_rate_ts2 * f32_hs,
            mp=mp,
            has_solid=has_solid_prec,
        )


def fes_equilibrium_clip(c, k, mp, dt, RATES, f=None):
    """
    Post-transport equilibrium clip for FeS precipitation.

    Called once per timestep AFTER the FiPy transport sweep has converged.
    Where Ω > 1, precipitates the minimum δ [mmol/L_pw] needed to restore Ω = 1.

    Equilibrium condition after clipping:
        (fe2_total - δ) · fe2_diss · (ts2 - δ) · hs_frac = H⁺ · Ksp

    This is a quadratic in δ (δ = moles precipitated per litre porewater):
        δ² - (fe2_total + ts2)·δ + (fe2_total·ts2 - H⁺·Ksp / (fe2_diss·hs_frac)) = 0

    The smaller positive root gives the minimum precipitation to reach equilibrium.

    Units:
        fe2_total, ts2  : mmol / L_pw
        fes             : mmol / L_solid
        δ               : mmol / L_pw
        fes_delta       : mmol / L_solid  =  δ · φ / (1 − φ)

    Note: dissolution (τ ≈ 0.3 yr) is not included here; add an explicit
    first-order step if under-prediction of FeS dissolution matters.
    """

    def update_rate(species, delta_val):
        if RATES is not None and species in RATES:
            RATES[species] = RATES[species] + delta_val
        if f is not None and hasattr(f, species):
            old_tuple = getattr(f, species)
            # old_tuple layout is (lhs_coeff, RHS[species], RATES[species], cross_term)
            # We replace RATES[species] with the updated one
            new_tuple = (old_tuple[0], old_tuple[1], RATES[species], old_tuple[3])
            setattr(f, species, new_tuple)

    # If we are not in the solver's clip step (e.g. we are in save_data_async or capture_state),
    # concentrations are already clipped. We just report the cached rates.
    if not getattr(mp, "in_clip", False):
        if RATES is not None and getattr(mp, "fes_clip_rate", None) is not None:
            rate_bulk = mp.fes_clip_rate
            update_rate("fe2_total", -rate_bulk)
            update_rate("ts2", -rate_bulk)
            update_rate("fes", rate_bulk)
        return

    # ── current values ────────────────────────────────────────────────────
    fe2_val = nx.maximum(c.fe2_total.value, 0.0)
    ts2_val = nx.maximum(c.ts2.value, 0.0)
    fes_val = nx.maximum(c.fes.value, 0.0)

    fe2_pw = fe2_val * mp.fe2_diss
    hs_val = ts2_val * mp.hs_frac
    omega = fe2_pw * hs_val / (k.hplus * k.fes_sp + 1e-30)

    needs_clip = omega > 1.0
    if not nx.any(needs_clip):
        mp.fes_clip_rate = nx.zeros_like(fe2_val)
        return

    # ── quadratic coefficients ────────────────────────────────────────────
    # δ² - (fe2_total + ts2)·δ + C = 0
    # where C = fe2_total·ts2 − H⁺·Ksp / (fe2_diss·hs_frac)
    ab = mp.fe2_diss * mp.hs_frac  # product of fractions
    target = (k.hplus * k.fes_sp) / (
        ab + 1e-30
    )  # equilibrium product in total-conc space

    b_coef = -(fe2_val + ts2_val)  # always negative
    c_coef = fe2_val * ts2_val - target  # positive where Ω > 1

    discriminant = nx.maximum(b_coef**2 - 4.0 * c_coef, 0.0)
    sqrt_disc = nx.sqrt(discriminant)

    # smaller root: minimum precipitation to reach equilibrium
    delta = (-b_coef - sqrt_disc) / 2.0

    # ── guard rails ───────────────────────────────────────────────────────
    delta = nx.maximum(delta, 0.0)
    delta = nx.minimum(
        delta, nx.minimum(fe2_val, ts2_val)
    )  # cannot exceed availability
    delta = nx.where(needs_clip, delta, 0.0)  # zero outside clip cells

    # ── convert and apply ─────────────────────────────────────────────────
    # δ is in mmol/L_pw; fes is stored in mmol/L_solid
    phi_val = getattr(mp.phi, "value", mp.phi)
    fes_delta_solid = delta * phi_val / (1.0 - phi_val + 1e-30)

    c.fe2_total.setValue(fe2_val - delta)
    c.ts2.setValue(ts2_val - delta)
    c.fes.setValue(fes_val + fes_delta_solid)
    i = 1000
    print(
        f"HS_bulk = {hs_val[i]:.2e}, Fe2_liq = {fe2_pw[i]:.2f}\n"
        f"fes value {fes_val[i]:.2e}, fes_delta = {fes_delta_solid[i]:.2e}, omega = {omega[i]:.2f}\n"
    )

    # ── calculate rates, cache them, and update RATES dict ────────────────
    rate_bulk = delta * phi_val / (dt + 1e-20)
    mp.fes_clip_rate = rate_bulk

    # print(f"Max clip rate = {nx.max(mp.fes_clip_rate):.2e} mmol/L/s")
    update_rate("fe2_total", -rate_bulk)
    update_rate("ts2", -rate_bulk)
    update_rate("fes", rate_bulk)
