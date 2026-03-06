"""Test iron mass conservation through reaction functions.

These tests verify that the net bulk iron production/consumption across all
iron species sums to zero for each reaction, ensuring mass conservation.

The 'divided form' conservation law states:
    phi * d(C_liquid)/dt + (1-phi) * d(C_solid)/dt = 0

This translates to requiring that the weighted sum of RATES for all iron
species equals zero for each reaction.
"""

import sys
import os

# Add the source directory to the path so we can import the modules
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src", "fipy-pyrite-burial")
)

import numpy as np
import pytest
from fipy import CellVariable
from fipy.meshes import Grid1D

from diff_lib import (
    add_implicit_coupling_new,
    add_implicit_sink,
    add_explicit_source,
    data_container,
)
from reactions_new import (
    sulfide_mediated_iron_reduction,
    fe2_oxidation,
    fes_oxidation,
    pyrite_oxidation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHI = 0.65  # porosity


def _make_mesh(n=5, dx=0.01):
    return Grid1D(nx=n, dx=dx)


def _make_mp():
    mp = data_container()
    mp.phi = PHI
    mp.fac_s = PHI / (1.0 - PHI)
    mp.eps = 1e-8
    mp.hs_frac = 0.24
    mp.h2s_frac = 0.76
    mp.fe2_pw_conc = 1.0 / (PHI + (1.0 - PHI) * 696)
    mp.f_sorb = (1.0 - PHI) * 696 / (PHI + (1.0 - PHI) * 696)
    mp.f_diss = PHI / (PHI + (1.0 - PHI) * 696)
    return mp


def _make_concentrations(mesh):
    c = data_container()
    species = {
        "so4": 28.0,
        "so4_32": 27.0,
        "ts2": 0.5,
        "ts2_32": 0.48,
        "o2": 0.1,
        "poc": 10.0,
        "fe2_total": 0.5,
        "fe3": 5.0,
        "fes": 0.1,
        "fes_32": 0.09,
        "s0": 0.1,
        "s0_32": 0.09,
        "fes2": 0.05,
        "fes2_32": 0.04,
        "fe2": 0.0,
        "fe2_p": 0.0,
        "h2s": 0.0,
        "h2s_32": 0.0,
        "hs": 0.0,
        "hs_32": 0.0,
        "hplus": 0.0,
    }
    for name, val in species.items():
        setattr(
            c,
            name,
            CellVariable(name=name, mesh=mesh, value=val, hasOld=True),
        )
    return c


def _make_k():
    k = data_container()
    k.fe3_hs = 1e-8
    k.fe2_ox = 1e-7
    k.fes_ox = 1e-8
    k.fes2_ox = 1e-10
    k.fes_s0 = 5e-8
    k.fes_ts2 = 5e-8
    k.poc_o2 = 5e-10
    k.poc_so4 = 1e-12
    k.hs_ox = 1e-7
    k.s0_ox = 4e-7
    k.hplus = 10 ** (-7.5)
    k.fes_sp = (1 - PHI) * 10 ** (-3.5)
    k.fes_isp = (1 - PHI) * 1e4
    k.fes_isd = (1 - PHI) * 3
    k.fe2_p_eq = 696
    return k


def _init_accumulators(c):
    species_list = list(c.keys())
    LHS = {s: np.zeros_like(c.so4.value) for s in species_list}
    CROSS = {s: [] for s in species_list}
    RHS = {s: np.zeros_like(c.so4) for s in species_list}
    RATES = {s: np.zeros_like(c.so4) for s in species_list}
    return LHS, CROSS, RHS, RATES


def _compute_bulk_iron_rate(RATES, phi):
    """Compute total bulk iron rate (mol/L_bulk/s).

    Iron species and their phases:
      fe2_total : dissolved  → weight by phi
      fe3       : solid      → weight by (1 - phi)
      fes       : solid      → weight by (1 - phi)
      fes2      : solid      → weight by (1 - phi)
    """
    fe2_bulk = phi * RATES["fe2_total"]
    fe3_bulk = (1 - phi) * RATES["fe3"]
    fes_bulk = (1 - phi) * RATES["fes"]
    fes2_bulk = (1 - phi) * RATES["fes2"]
    return fe2_bulk + fe3_bulk + fes_bulk + fes2_bulk


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSulfideMediatedIronReduction:
    """0.5 HS- + Fe3+ -> 0.5 S0 + Fe2+

    Iron budget: 1 Fe3 consumed → 1 Fe2 produced → net zero.
    """

    def test_iron_conservation(self):
        mesh = _make_mesh()
        mp = _make_mp()
        c = _make_concentrations(mesh)
        k = _make_k()
        lim = {}

        LHS, CROSS, RHS, RATES = _init_accumulators(c)
        sulfide_mediated_iron_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp)

        total = _compute_bulk_iron_rate(RATES, PHI)
        np.testing.assert_allclose(total, 0.0, atol=1e-20)


class TestFe2Oxidation:
    """4 Fe2+ + O2 -> Fe3OOH

    Iron budget: 1 Fe2 consumed → 1 Fe3 produced → net zero.
    """

    def test_iron_conservation(self):
        mesh = _make_mesh()
        mp = _make_mp()
        c = _make_concentrations(mesh)
        k = _make_k()
        lim = {}

        LHS, CROSS, RHS, RATES = _init_accumulators(c)
        fe2_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp)

        total = _compute_bulk_iron_rate(RATES, PHI)
        np.testing.assert_allclose(total, 0.0, atol=1e-20)


class TestFesOxidation:
    """FeS + 2.25 O2 -> Fe3 + SO4

    Iron budget: 1 FeS consumed → 1 Fe3 produced → net zero.
    """

    def test_iron_conservation(self):
        mesh = _make_mesh()
        mp = _make_mp()
        c = _make_concentrations(mesh)
        k = _make_k()
        lim = {}

        LHS, CROSS, RHS, RATES = _init_accumulators(c)
        fes_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp)

        total = _compute_bulk_iron_rate(RATES, PHI)
        np.testing.assert_allclose(total, 0.0, atol=1e-20)


class TestPyriteOxidation:
    """FeS2 + 3.5 O2 -> Fe3 + 2 SO4

    Iron budget: 1 FeS2 consumed → 1 Fe3 produced → net zero.
    """

    def test_iron_conservation(self):
        mesh = _make_mesh()
        mp = _make_mp()
        c = _make_concentrations(mesh)
        k = _make_k()
        lim = {}

        LHS, CROSS, RHS, RATES = _init_accumulators(c)
        pyrite_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp)

        total = _compute_bulk_iron_rate(RATES, PHI)
        np.testing.assert_allclose(total, 0.0, atol=1e-20)
