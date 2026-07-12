# Auto-generated test output

from fipyrite.diff_lib import add_coupled_reaction, add_implicit_sink, calculate_fractionated_coeff_32, partition_equilibrium_isotope_32

def sulfate_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = True
    poc_species = k.get('poc_species', 'POC_fast')
    poc_k_name = k.get('poc_k', 'POC_fast')
    k_val = mp.k.get(poc_k_name) if hasattr(mp, 'k') else k.get(poc_k_name, 0.0)
    rate_base = k_val * getattr(c, poc_species) * c.SO4 * lim['O2_inhibit'] * lim['SO4_implicit'] * lim['Fe3_diss_red_inhib']
    rate_master = 1 * rate_base
    coeff_master = rate_master / (c.SO4 + 1e-30)
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species='SO4',
        reactants={poc_species: 2},
        products={'TS2': 1},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='sulfate_reduction',
        ref_species=poc_species,
    )
    coeff_POC = rate_base / (getattr(c, poc_species) + 1e-30)
    add_implicit_sink(LHS, RATES, poc_species, coeff_POC, rate_base, mp=mp, has_solid=has_solid, c=c)
    if mp.isotopes:
        alpha = 1.0 + (mp.msr_alpha - 1.0) * lim['SO4_alpha_explicit']
        coeff_master_32 = calculate_fractionated_coeff_32(
            coeff_master, c.SO4, c.SO4_32, alpha, eps=1e-30
        )
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species='SO4_32',
            reactants={poc_species: 2},
            products={'TS2_32': 1},
            coeff_master=coeff_master_32,
            rate_master=coeff_master_32 * c.SO4_32,
            has_solid=has_solid,
            reaction_name='sulfate_reduction_32',
            ref_species=poc_species,
        )
