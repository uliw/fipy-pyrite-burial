# Considerations for Stable Isotope Modeling in FiPy

Modeling stable isotope fractionation (e.g., $^{32}\text{S}$ and $^{34}\text{S}$) in advection-diffusion-reaction equations with FiPy requires careful attention to reaction stoichiometry, matrix coupling, and near-zero numerical limits. Below is a guide to the key pitfalls discovered during development and their solutions.

---

## 1. Double-Counting Reactant Consumption

### The Pitfall
In diagenetic models, isotope reactions (e.g., $^{32}\text{S}$ reduction) are often solved as separate equations alongside the bulk species (which represents total S). If non-isotope reactants (like organic carbon $\text{POC}$ or solid $\text{Fe}^{3+}$) are included in the reactants list of both the bulk reaction and the isotope reaction:
```python
# BULK: 2 POC + SO4 -> TS2
add_coupled_reaction(..., master_species="SO4", reactants={"POC": 2.0}, ...)

# ISOTOPE: 2 POC + SO4_32 -> TS2_32 (INCORRECT COUPLING)
add_coupled_reaction(..., master_species="SO4_32", reactants={"POC": 2.0}, ...)
```
The solver adds off-diagonal matrix coupling terms to the $\text{POC}$ equation from both reactions. Since $^{32}\text{S}$ represents ~95% of total sulfur, $\text{POC}$ will be consumed at nearly **twice** its physical rate. This drives $\text{POC}$ negative, flipping the sign of its reaction terms, creating runaway positive feedback loops that crash the coupled solver.

### The Solution
Exclude bulk/non-isotope species from the reactants and products lists of the isotope-specific equations. Their consumption/production is already fully accounted for by the bulk reaction:
```python
# ISOTOPE: (CORRECT COUPLING)
add_coupled_reaction(..., master_species="SO4_32", reactants={}, products={"TS2_32": 1.0}, ...)
```

---

## 2. Stoichiometric Scaling Mismatches

### The Pitfall
When you remove non-isotope reactants from the reactants list of the isotope reaction to prevent double-counting, the reference species (e.g., $\text{POC}$ in sulfate reduction) is no longer available in the local `reactants` list of the isotope reaction call. 
* If the scaling factor helper determines the reference species stoichiometry by checking the `reactants` dict, it will fail to find it and fall back to a default stoichiometry of `1.0`.
* This causes the bulk reaction rate to be scaled by `stoich_master / stoich_ref = 1.0 / 2.0 = 0.5`, while the isotope reaction is scaled by `1.0 / 1.0 = 1.0`.
* As a result, the isotope reaction runs at **twice the relative rate** of the bulk reaction, driving the isotope down too fast and creating highly skewed isotope ratios (e.g., $\delta^{34}\text{S} > 100\text{ mUr}$ when it should be $< 40\text{ mUr}$).

### The Solution
Explicitly pass the original reaction's reference stoichiometry (`stoich_ref`) to the matrix coupling helper so that it does not rely on local `reactants` dictionary lookups:
```python
add_coupled_reaction(
    ...,
    master_species={"SO4_32": 1},
    reactants={},
    products={"TS2_32": 1},
    stoich_ref=2.0,  # Explicitly matches the bulk reaction's reference stoichiometry
)
```

---

## 3. Isotope Ratio Numerical Noise near Zero

### The Pitfall
Isotope rate coefficients are calculated using the non-linear denominator correction term:
$$\text{denom\_ratio} = 1.0 + (\alpha - 1.0) \cdot \frac{C_{32}}{C_{total}}$$
At high-consumption boundaries (e.g., where sulfate is depleted near zero), the ratio $\frac{C_{32}}{C_{total}}$ becomes highly sensitive to tiny floating-point truncation errors (since you are dividing two extremely small numbers). This numerical noise makes the matrix stiff and causes solver convergence failures.

### The Solution
Implement a concentration threshold (e.g., $1\ \mu\text{mol/L}$ or $10^{-3}\ \text{mmol/L}$) in the fractionation coefficient calculation. If the total concentration falls below this threshold, fractionation is bypassed ($\alpha_{eff} = 1.0$), neutralizing the non-linear denominator correction and stabilizing the solver:
```python
def calculate_fractionated_coeff_32(coeff_total, c_total, c_32, alpha, eps=1e-20):
    c_tot_np = np.asarray(c_total)
    
    # Neglect isotope fractionation effects below 1 umol/L
    alpha_eff = np.where(c_tot_np > 1e-3, alpha, 1.0)
    
    ratio_32 = np.where(c_tot_np > 1e-15, np.asarray(c_32) / (c_tot_np + 1e-30), 0.95770423)
    ratio_32 = np.clip(ratio_32, 0.5, 1.5)
    denom_ratio = 1.0 + (alpha_eff - 1.0) * ratio_32
    return coeff_total * alpha_eff / denom_ratio
```
