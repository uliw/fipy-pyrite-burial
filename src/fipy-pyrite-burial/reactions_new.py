"""Define the reactions."""

from __future__ import annotations

import numpy as np
import fipy as fp
from typing import Callable, Dict, Mapping, Tuple

from diff_lib import calculate_k_iron_reduction

# ------------------------------------------------------------
# Short aliases – they make the signatures easier to read
# ------------------------------------------------------------
Var = fp.CellVariable  # a FiPy field (concentration, solid mass …)
Rate = float | np.ndarray  # either a scalar or a full‑mesh array
Coeff = float | np.ndarray  # same as Rate


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
    for s in species_list:
        setattr(f, s, (LHS[s], RHS[s], RATES[s], CROSS[s]))

    return f, RATES


# =============================================================================
# HELPER FUNCTIONS (Matrix Math Abstraction)
# =============================================================================


# ----------------------------------------------------------------------
# Helper – make sure LHS[species] is a list of ImplicitSourceTerm objects
# ----------------------------------------------------------------------
def _ensure_term_list(
    LHS: Dict[str, Union[float, List[fp.ImplicitSourceTerm]]],
    species: str,
) -> List[fp.ImplicitSourceTerm]:
    """
    Return a list that will hold the implicit terms for *species*.
    If the entry is still a plain float (old style), convert it to a single
    ``ImplicitSourceTerm`` with that coefficient and replace the entry with a
    list containing the term.
    """
    entry = LHS.get(species, None)

    # --------------------------------------------------------------
    # 1️⃣  No entry yet → create an empty list
    # --------------------------------------------------------------
    if entry is None:
        LHS[species] = []  # type: ignore[assignment]
        return LHS[species]  # type: ignore[return-value]

    # --------------------------------------------------------------
    # 2️⃣  Entry already a list of terms → just return it
    # --------------------------------------------------------------
    if isinstance(entry, list):
        # we know the list only contains ImplicitSourceTerm objects
        return entry

    # --------------------------------------------------------------
    # 3️⃣  Entry is a plain numeric coefficient → wrap it
    # --------------------------------------------------------------
    if isinstance(entry, (float, int, np.ndarray)):
        # The old code stored the *diagonal* Jacobian entry.
        # We create a term with that coefficient; the variable will be
        # attached later (see the solver set‑up routine).
        term = fp.ImplicitSourceTerm(var=None, coeff=entry)
        LHS[species] = [term]  # replace the scalar with a list
        return LHS[species]  # type: ignore[return-value]

    # --------------------------------------------------------------
    # 4️⃣  Anything else is a programming error
    # --------------------------------------------------------------
    raise TypeError(
        f"LHS['{species}'] has unsupported type {type(entry)}. "
        "It must be a float, numpy.ndarray, or a list of ImplicitSourceTerm."
    )


# ----------------------------------------------------------------------
# Revised add_implicit_sink – works with both legacy and new style
# ----------------------------------------------------------------------
def add_implicit_sink(
    LHS: Dict[str, Union[float, List[fp.ImplicitSourceTerm]]],
    RATES: Dict[str, float],
    species: str,
    coeff: float,
    rate: float,
) -> None:
    """
    Append a linear sink ``‑coeff·var`` to the equation for *species*.

    Parameters
    ----------
    LHS
        Dictionary that stores either a plain numeric Jacobian entry
        (legacy) or a list of ``ImplicitSourceTerm`` objects (new).
    RATES
        Diagnostic container – the sum of the *raw* (non‑linear) rates.
    species
        Name of the variable that receives the sink.
    coeff
        Linear coefficient (positive number).  The function will prepend the
        required minus sign, i.e. the term that is added to the matrix is
        ``‑coeff``.
    rate
        The non‑linear rate value (used only for bookkeeping).
    """
    # ------------------------------------------------------------------
    # 1️⃣  Make sure we have a list of terms for this species
    # ------------------------------------------------------------------
    term_list = _ensure_term_list(LHS, species)

    # ------------------------------------------------------------------
    # 2️⃣  Build the new sink term and append it
    # ------------------------------------------------------------------
    # The sign is negative because this is a *sink*.
    new_term = fp.ImplicitSourceTerm(var=None, coeff=-coeff)
    term_list.append(new_term)

    # ------------------------------------------------------------------
    # 3️⃣  Update the diagnostic RATES dictionary
    # ------------------------------------------------------------------
    RATES[species] = RATES.get(species, 0.0) + float(np.asarray(rate).sum())


# ----------------------------------------------------------------------
# NEW add_implicit_coupling ------------------------------------------------
# ----------------------------------------------------------------------
def add_implicit_coupling(
    LHS: Dict[str, Union[float, List[fp.ImplicitSourceTerm]]],
    CROSS: Dict[str, List[Tuple[str, float]]],
    RATES: Dict[str, float],
    target_species: str,
    source_species: str,
    coeff: float,
    rate: float,
) -> None:
    """
    Record that *target_species* receives a linear source from *source_species*:

        d[target]/dt  =  + coeff · [source]

    The function does three things:

    1. Append the pair ``(source_species, coeff)`` to the ``CROSS`` dictionary
       – this is the data structure you already use later when you build the
       full FiPy equation.

    2. Ensure that ``LHS[target_species]`` is a **list of**
       :class:`fipy.terms.implicitSourceTerm.ImplicitSourceTerm` objects.
       If the entry is still a legacy numeric Jacobian entry we wrap it
       automatically with ``_ensure_term_list``.

    3. Append a *new* ``ImplicitSourceTerm`` that represents the coupling.
       The term is created with ``var=None`` for now; the real source variable
       will be attached in the solver‑setup routine (the same place where you
       attach the variable to the precipitation terms).

    Parameters
    ----------
    LHS
        Dictionary that stores either a plain numeric Jacobian entry (legacy)
        or a list of ``ImplicitSourceTerm`` objects (new).  It is mutated in‑place.
    CROSS
        ``{target: [(source, coeff), …]}`` – kept for backward compatibility.
    RATES
        Diagnostic container; we simply accumulate the raw rate value.
    target_species, source_species
        Names of the dependent (target) and independent (source) variables.
    coeff
        Linear coefficient that multiplies the source concentration.
    rate
        The (non‑linear) rate value – stored only for reporting.
    """
    # --------------------------------------------------------------
    # 1️⃣  Record the coupling in the CROSS dict (unchanged)
    # --------------------------------------------------------------
    CROSS.setdefault(target_species, []).append((source_species, coeff))

    # --------------------------------------------------------------
    # 2️⃣  Make sure LHS[target] is a list of terms (convert legacy scalar)
    # --------------------------------------------------------------
    term_list = _ensure_term_list(LHS, target_species)

    # --------------------------------------------------------------
    # 3️⃣  Build the coupling term and append it
    # --------------------------------------------------------------
    # The term represents  +coeff * source_variable .
    # We keep var=None for now – it will be replaced by the actual
    # ``CellVariable`` of *source_species* just before the solve.
    coupling_term = fp.ImplicitSourceTerm(var=None, coeff=coeff)
    term_list.append(coupling_term)

    # --------------------------------------------------------------
    # 4️⃣  Update the diagnostic RATES dictionary
    # --------------------------------------------------------------
    RATES[target_species] = RATES.get(target_species, 0.0) + float(
        np.asarray(rate).sum()
    )


# ----------------------------------------------------------------------
# Revised add_implicit_coupling_new
# ----------------------------------------------------------------------
def add_implicit_coupling_new(
    ctype: str,
    CROSS: Dict[str, List[Tuple[str, float]]],
    RATES: Dict[str, float],
    LHS: Dict[str, Union[float, List[fp.ImplicitSourceTerm]]],
    target_species: str,
    source_species: str,
    coeff: float,
    rate: float,
    mp: Dict[str, float],
) -> None:
    """
    Add a *linear* coupling between two species while honouring porosity
    corrections.

    The reaction is interpreted as

        d[target]/dt =  + (coeff·fac) · source

    where ``fac`` depends on the connection type (`ctype`).  The routine

    1. records the off‑diagonal entry in ``CROSS`` (used later when the
       global matrix is assembled);
    2. appends an ``ImplicitSourceTerm`` to the *target* equation;
    3. adds the corresponding sink to the *source* equation (via the
       already‑robust ``add_implicit_sink`` helper);
    4. updates the diagnostic ``RATES`` dictionary.

    Parameters
    ----------
    ctype
        Connection type – one of ``'liquid_2_liquid'``,
        ``'liquid_2_solid'``, ``'solid_2_solid'``,
        ``'solid_2_liquid'``.
    CROSS
        Off‑diagonal coupling container (``{target: [(source, coeff), …]}``).
    RATES
        Diagnostic dictionary that stores the summed raw rate for each species.
    LHS
        Diagonal (implicit‑sink) container.  Each entry may be a plain
        ``float`` (legacy) **or** a ``list[ImplicitSourceTerm]`` (new style).
    target_species, source_species
        Names of the variables that appear on the left‑ and right‑hand side
        of the coupling, respectively.
    coeff
        Linear coefficient (positive number) *before* porosity correction.
    rate
        The raw (non‑linear) rate value – used only for bookkeeping.
    mp
        Model‑parameter dictionary that must contain ``fac_s`` (the porosity
        factor ``phi/(1‑phi)``).
    """
    # --------------------------------------------------------------
    # 1️⃣  Porosity correction factor
    # --------------------------------------------------------------
    if ctype == "liquid_2_liquid":
        fac = 1.0
    elif ctype == "liquid_2_solid":
        fac = mp["fac_s"]
    elif ctype == "solid_2_solid":
        fac = 1.0
    elif ctype == "solid_2_liquid":
        fac = 1.0 / mp["fac_s"]
    else:
        raise ValueError(
            f"ctype must be one of "
            f"liquid_2_liquid, liquid_2_solid, solid_2_solid, solid_2_liquid; got {ctype}"
        )

    # --------------------------------------------------------------
    # 2️⃣  Record the off‑diagonal entry (used later when we build the matrix)
    # --------------------------------------------------------------
    CROSS.setdefault(target_species, []).append((source_species, coeff * fac))

    # --------------------------------------------------------------
    # 3️⃣  Append an ImplicitSourceTerm to the *target* equation
    # --------------------------------------------------------------
    term_list_target = _ensure_term_list(LHS, target_species)
    term_target = fp.ImplicitSourceTerm(var=None, coeff=coeff * fac)
    term_list_target.append(term_target)

    # --------------------------------------------------------------
    # 4️⃣  Add the corresponding sink on the *source* species
    # --------------------------------------------------------------
    # The sink uses the *un‑scaled* coefficient (the consumption rate of the
    # source species).  ``add_implicit_sink`` will automatically turn a legacy
    # scalar into a term if necessary.
    add_implicit_sink(LHS, RATES, source_species, coeff, rate)

    # --------------------------------------------------------------
    # 5️⃣  Update diagnostic rate bookkeeping for the target species
    # --------------------------------------------------------------
    RATES[target_species] = (
        RATES.get(target_species, 0.0) + getattr(rate, "value", rate) * fac
    )


def add_explicit_source(RHS, RATES, species, rate):
    """Add a production term to the RHS vector.

    Add a production term to the RHS vector.
    RHS = -Rate (Standard library quirk for production)
    """
    RHS[species] = RHS[species] + rate
    RATES[species] += getattr(rate, "value", rate)


def apply_rate_limiter(rate, var, fraction=0.5, eps=1e-12):
    """Limit rate so it doesn't consume more than a fraction of available var."""
    val = var.value if hasattr(var, "value") else var
    max_rate = val * fraction / 1.0  # Normalized dt=1 for steady state sweep
    return np.minimum(rate, np.maximum(max_rate, 0.0))


# =============================================================================
# PROCESS FUNCTIONS (The Biogeochemistry)
# =============================================================================
def aerobic_respiration(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Define POC consumption by aerobic respiration."""
    rate_base = k.poc_o2 * c.poc * c.o2

    # POC Sink - SOLID
    coeff_poc = k.poc_o2 * c.o2 * mp.fac_s
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, rate_base * mp.fac_s)

    # O2 Sink (1.27x) - LIQUID
    coeff_o2 = 1.27 * k.poc_o2 * c.poc
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, 1.27 * rate_base)
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
    add_implicit_sink(LHS, RATES, "poc", coeff_poc, poc_rate * mp.fac_s)

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
        mp,  # model parameters
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
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2)

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
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2)

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
        )


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
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, rate_base * 0.25)

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
    )

    # O2 Sink (2.25x) - LIQUID
    # Depends on FeS (Solid).
    coeff_o2_fes = 2.25 * k.fes_ox * c.fes
    rate_base = k.fes_ox * c.fes * c.o2
    # Implicit Sink for O2: coeff = 2.25 * k * FeS.
    add_implicit_sink(LHS, RATES, "o2", coeff_o2_fes, rate_base * 2.25)

    # SO4 Source (1.0x) - LIQUID
    # Couple to FeS.
    # Target Liquid. No mp.fac_s.
    add_implicit_coupling(
        LHS,
        CROSS,
        RATES,
        "so4",
        "fes",
        k.fes_ox * c.o2,
        rate_base,
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
        )


def pyrite_formation_s0(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 1 S0 -> 1 FeS2.
    This is a bit tricky as we have two different S atoms into the same
    target (FeS2)
    """
    # S0 Sink Solid
    coeff_s0 = k.fes_s0 * c.fes * mp.fac_s
    add_implicit_sink(LHS, RATES, "s0", coeff_s0, coeff_s0 * c.s0)

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
        )


def pyrite_formation_fes_ts2(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """reaction: 1 FeS + 1 H2S -> 1 FeS2"""
    # FeS Sink - SOLID
    coeff_fes = k.fes_ts2 * c.ts2 * mp.fac_s
    add_implicit_sink(LHS, RATES, "fes", coeff_fes, coeff_fes * c.fes)
    add_implicit_sink(LHS, RATES, "fes_32", coeff_fes, coeff_fes * c.fes_32)

    # H2S Sink (1.0x) - LIQUID
    coeff_ts2 = k.fes_ts2 * c.fes
    add_implicit_sink(LHS, RATES, "ts2", coeff_ts2, coeff_ts2 * c.ts2)
    add_implicit_sink(LHS, RATES, "ts2_32", coeff_ts2, coeff_ts2 * c.ts2_32)

    # FeS2 Source (1.0x) - SOLID
    # Couple to FeS.
    # Rate = k * H2S * FeS.
    add_implicit_coupling(
        LHS,
        CROSS,
        RATES,
        "fes2",
        "fes",
        k.fes_ts2 * c.ts2 * mp.fac_s,
        k.fes_ts2 * c.ts2 * c.fes * mp.fac_s,
    )

    # FeS2_32 Source
    # From FeS_32 and H2S_32.
    # Couple to FeS_32
    add_implicit_coupling(
        LHS,
        CROSS,
        RATES,
        "fes2_32",
        "fes_32",
        k.fes_ts2 * c.ts2 * mp.fac_s,
        k.fes_ts2 * c.ts2 * c.fes_32 * mp.fac_s,
    )
    # Couple to H2S_32
    # Rate = k * FeS * H2S_32.
    # Target Solid, Source Liquid. Coeff Needs mp.fac_s.
    add_implicit_coupling(
        LHS,
        CROSS,
        RATES,
        "fes2_32",
        "ts2_32",
        mp.fac_s * k.fes_ts2 * c.fes,
        mp.fac_s * k.fes_ts2 * c.fes * c.ts2_32,
    )


def pyrite_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Reaction: 1 FeS2 + 3.5 O2 -> 1 Fe3 + 2 SO4
    Ref: FeS2 (k.fes2_ox)
    """
    # FeS2 Sink - SOLID
    coeff_fes2 = k.fes2_ox * c.o2 * mp.fac_s
    add_implicit_sink(LHS, RATES, "fes2", coeff_fes2, coeff_fes2 * c.fes2)
    add_implicit_sink(LHS, RATES, "fes2_32", coeff_fes2, coeff_fes2 * c.fes2_32)

    # O2 Sink (3.5x) - LIQUID
    coeff_o2 = 3.5 * k.fes2_ox * c.fes2
    add_implicit_sink(LHS, RATES, "o2", coeff_o2, coeff_o2 * c.o2)

    # Fe3 Source (1.0x) - SOLID
    # Couple to FeS2
    add_implicit_coupling(
        LHS,
        CROSS,
        RATES,
        "fe3",
        "fes2",
        k.fes2_ox * c.o2 * mp.fac_s,
        k.fes2_ox * c.o2 * c.fes2 * mp.fac_s,
    )

    # SO4 Source (2.0x) - LIQUID
    # Couple to FeS2
    add_implicit_coupling(
        LHS,
        CROSS,
        RATES,
        "so4",
        "fes2",
        2 * k.fes2_ox * c.o2,
        2 * k.fes2_ox * c.o2 * c.fes2,
    )
    # SO4_32
    add_implicit_coupling(
        LHS,
        CROSS,
        RATES,
        "so4_32",
        "fes2_32",
        k.fes2_ox * c.o2,
        k.fes2_ox * c.o2 * c.fes2_32,
    )


def fes_formation_fully_implicit_2(
    c: Mapping[str, fp.CellVariable],
    k: Mapping[str, float],
    lim: Mapping[str, float],
    LHS: Dict[str, fp.ImplicitSourceTerm],
    RHS: Dict[str, fp.ExplicitSourceTerm],
    RATES: Dict[str, float],
    CROSS: Dict[str, List[Tuple[str, float]]],
    mp: Mapping[str, float],
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
    fe2: fp.CellVariable = c["fe2_total"]
    hs: fp.CellVariable = c["ts2"]
    fes: fp.CellVariable = c["fes"]

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
    term_fe2 = fp.ImplicitSourceTerm(var=fe2, coeff=-rate_precip * mp["f_diss"])
    if "fe2_total" in LHS:
        LHS["fe2_total"].coeff += term_fe2.coeff
    else:
        LHS["fe2_total"] = term_fe2

    # ---- H₂S --------------------------------------------------------
    term_hs = fp.ImplicitSourceTerm(var=hs, coeff=-rate_precip)
    if "ts2" in LHS:
        LHS["ts2"].coeff += term_hs.coeff
    else:
        LHS["ts2"] = term_hs

    # ---- FeS (solid) -------------------------------------------------
    term_fes = fp.ImplicitSourceTerm(var=fes, coeff=rate_precip * mp["fac_s"])
    if "fes" in LHS:
        LHS["fes"].coeff += term_fes.coeff
    else:
        LHS["fes"] = term_fes

    # -----------------------------------------------------------------
    # 3️⃣  Book‑keeping of the (non‑linear) precipitation rate
    # -----------------------------------------------------------------
    # ``RATES`` is only for diagnostics – we store the *cell‑averaged* value.
    avg_rate = float(rate_precip.mean())
    RATES["fe2_total"] = RATES.get("fe2_total", 0.0) + avg_rate
    RATES["ts2"] = RATES.get("ts2", 0.0) + avg_rate
    RATES["fes"] = RATES.get("fes", 0.0) + avg_rate

    # -----------------------------------------------------------------
    # 4️⃣  Slow dissolution (still linear, therefore we keep the helpers)
    # -----------------------------------------------------------------
    #   Ω_diss = min(Ω, 1)   →  (1‑Ω_diss) is the driving force
    omega_diss = fp.minimum(omega, 1.0)  # FiPy expression
    diss_mask = 1.0 - fp.where(omega > 1.0, 1.0, 0.0)  # same as 1‑precip_mask

    fes_diss_coeff = (
        k["fes_isd"] * (1.0 - omega_diss) * diss_mask * lim["fes_explicit"]
    )  # this is a *linear* coefficient that multiplies the solid

    # Fe²⁺ source from dissolution
    add_implicit_coupling(
        LHS,
        CROSS,
        RATES,
        target_species="fe2_total",
        source_species="fes",
        coeff=fes_diss_coeff,
        rate=fes_diss_coeff * fes.value,
    )

    # H₂S source from dissolution
    add_implicit_coupling(
        LHS,
        CROSS,
        RATES,
        target_species="ts2",
        source_species="fes",
        coeff=fes_diss_coeff,
        rate=fes_diss_coeff * fes.value,
    )

    # FeS sink from dissolution (multiply by the solid‑phase factor)
    coeff_fes_sink = fes_diss_coeff * mp["fac_s"]
    add_implicit_sink(LHS, RATES, "fes", coeff_fes_sink, coeff_fes_sink * fes.value)

    # -----------------------------------------------------------------
    # 5️⃣  Optional isotope bookkeeping (unchanged – still uses helpers)
    # -----------------------------------------------------------------
    if "ts2_32" in c:
        ts2_32: fp.CellVariable = c["ts2_32"]
        fes_32: fp.CellVariable = c["fes_32"]

        # 5a – precipitation (non‑linear part) for 32S
        ts2_inv = 1.0 / (hs.value + 1e-20)

        coeff_ts2_32_precip = (
            k["fes_isp"]
            * (fe2 * mp["f_diss"] / omega_den - ts2_inv)
            * fp.where(omega > 1.0, 1.0, 0.0)
        )
        coeff_ts2_32_precip = fp.maximum(coeff_ts2_32_precip, 0.0)

        add_implicit_sink(
            LHS,
            RATES,
            "ts2_32",
            coeff_ts2_32_precip,
            coeff_ts2_32_precip * ts2_32.value,
        )

        # 5b – coupling of 32S to the solid
        add_implicit_coupling(
            LHS,
            CROSS,
            RATES,
            target_species="fes_32",
            source_species="ts2_32",
            coeff=coeff_ts2_32_precip * mp["fac_s"],
            rate=coeff_ts2_32_precip * ts2_32.value * mp["fac_s"],
        )

        # 5c – dissolution sink on the solid (same coeff as for the main solid)
        add_implicit_sink(
            LHS,
            RATES,
            "fes_32",
            coeff_fes_sink,
            coeff_fes_sink * fes_32.value,
        )

        # 5d – dissolution source on H₂S_32 (coupled to solid)
        add_implicit_coupling(
            LHS,
            CROSS,
            RATES,
            target_species="ts2_32",
            source_species="fes_32",
            coeff=fes_diss_coeff,
            rate=fes_diss_coeff * fes_32.value,
        )
