# Fipyrite Reaction Coupling Refactoring Report

We have successfully introduced the `add_coupled_reaction` wrapper function in [diff_lib.py](file:///home/uliw/user/python-scripts/rtm_modeling/fipy/fipyrite/src/fipyrite/diff_lib.py) and refactored the straightforward reaction functions in [reactions_new.py](file:///home/uliw/user/python-scripts/rtm_modeling/fipy/fipyrite/experiments/reactions_new.py).

Below is a summary of the changes made, followed by an analysis of the complex cases that were intentionally skipped.

---

## 1. Summary of Refactoring Changes

The new wrapper `add_coupled_reaction` has been added to [diff_lib.py:L1194](file:///home/uliw/user/python-scripts/rtm_modeling/fipy/fipyrite/src/fipyrite/diff_lib.py#L1194). This wrapper manages:
- Applying the self-implicit sink term on the `master_species`.
- Appending cross-coupling terms to `CROSS` for both reactants and products.
- Handing porosity conversions using the `has_solid` boolean.
- Correctly tracking and updating both bulk rates and reaction-specific rates in the `RATES` dictionary.

The following **11 functions** were successfully and cleanly refactored in [reactions_new.py](file:///home/uliw/user/python-scripts/rtm_modeling/fipy/fipyrite/experiments/reactions_new.py):

*   **`dissimilatory_iron_reduction`**
*   **`sulfate_reduction`**
*   **`hs_oxidation`**
*   **`hs_oxidation_velde`**
*   **`elemental_sulfur_oxidation`**
*   **`sulfide_mediated_iron_reduction`**
*   **`pyrite_formation_s0`**
*   **`pyrite_formation_fes_ts2`**
*   **`pyrite_oxidation`**
*   **`fes_dissolution`**
*   **`s0_disproportionation`**

For these functions, the code length was significantly reduced, and stoichiometry is now expressed declaratively (e.g. `reactants={}` and `products={"so4": so4_fraction, "ts2": h2s_fraction}`).

---

## 2. Skipped/Complex Cases

As requested, any functions that involved complex mathematical structures, multiple coupled master variables, or explicit target relaxation schemes were left unchanged. Below is the detailed reasoning and potential path forward for each.

### A. `aerobic_respiration`
*   **Reason for skipping:** This function does not perform any off-diagonal cross-coupling (`CROSS` remains empty). It consists of two uncoupled, diagonal self-implicit sinks: one for `poc_fast`/`poc_slow` (solid), and one for `o2` (dissolved).
*   **Recommendation:** Keep as-is. Since there is no coupling between different species at the matrix level, using the wrapper is unnecessary.

### B. Obsolete / Buggy Sulfide-Mediated Reductions
*   **`sulfide_mediated_iron_reduction_old`**: This is an obsolete version that is no longer active in the model.
*   **`sulfide_mediated_iron_reduction_velde`**:
    *   **Reason for skipping:** In lines 794 and 809, the original code registers the TS2 sink but sets `target_species="so4"` twice, suggesting a copy-paste bug in the original code. 
    *   **Recommendation:** To preserve exact numeric reproducibility of any legacy runs, we did not modify this. If Velde's reduction is needed in the future, this typo should be corrected first.

### C. `fes_precipitation`
*   **Reason for skipping:** This function uses a complex relaxation scheme targeting a quadratic-bounded concentration equilibrium (`ts2_target`). The rate is:
    $$\frac{d[\text{ts2}]}{dt} = -k_{\text{rxn\_eff}} \cdot [\text{ts2}] + k_{\text{rxn\_eff}} \cdot [\text{ts2\_target}]$$
    This is solved as a hybrid term: an implicit sink proportional to `ts2` combined with an explicit source proportional to `ts2_target` (added via `add_explicit_source` on the RHS). Because our wrapper only manages the fully implicit part, trying to force this into the wrapper would require separate manual `add_explicit_source` calls, which reduces the readability benefits.

### D. `fes_precipitation_terminal`
*   **Reason for skipping:** This function uses an asymmetric coupling structure where:
    - `fes` is cross-coupled to `fe2_total` (acting as the master).
    - `ts2` is solved with a separate self-implicit diagonal sink (acting as its own master).
    This means the precipitation process is not unified under a single master variable, which violates the standard single-master assumption of `add_coupled_reaction`.

### E. `fes_precipitation_dissolution_linearized`
*   **Reason for skipping:** This function performs a manual multi-variable Newton-Raphson linearization of the saturation ratio $\Omega$ around the current sweep iterate. Because it requires taking partial derivatives with respect to multiple variables simultaneously:
    - It adds diagonal implicit terms.
    - It adds off-diagonal cross-coupling terms for multiple variables.
    - It adds an explicit correction term to the RHS.
    This level of mathematical complexity is too bespoke for the general-purpose `add_coupled_reaction` wrapper.
