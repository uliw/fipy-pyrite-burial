import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fipy import CellVariable, Grid1D
from fipyrite.diff_lib import data_container
from fipyrite.solver_calls import run_non_steady_state_solver_coupled

class MockMP:
    def __init__(self):
        self.plot_name = "test_run"
        self.max_steps = 3
        self.start_time = 0.0
        self.t_end = 100.0
        self.dt_min = 1.0
        self.dt_init = 5.0
        self.dt_max = 50.0
        self.backup_step = 100
        self.report_step = 100
        self.isotopes = False
        self.title = "test"
        self.enable_rate_adaptation = True
        self.monitored_rate_species = ["FeS", "TS2"]
        self.rate_threshold = 1e-8
        self.phi = 0.8
        self.solver_backend = "default"
        self.tolerance = 1e-12
        self.dt_tolerance = -1.0
        self.adaptive_solver_tolerance = False
        self.rate_adaptation_start_step = 2
        self.enable_rate_magnitude_check = True

class MockD:
    def __init__(self):
        self.FeS = 0.0
        self.TS2 = 0.0
        self.D_bio = 0.0
        self.D_irr = 0.0

@pytest.fixture
def solver_inputs():
    mesh = Grid1D(nx=3)
    c = data_container({
        "FeS": CellVariable(mesh=mesh, value=1.0, hasOld=True),
        "TS2": CellVariable(mesh=mesh, value=1.0, hasOld=True),
    })
    mp = MockMP()
    k = data_container()
    D_mol = MockD()
    bc_map = {
        "FeS": {"type": "solid", "top": 0.0},
        "TS2": {"type": "dissolved", "top": 0.0}
    }
    z = np.array([0.0, 1.0, 2.0, 3.0])
    
    return mp, c, k, mesh, D_mol, bc_map, z

@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_adaptation_success(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that a step with valid rate changes is accepted and dt grows."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    
    # Mock setup static equations
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    # Configure mock update rates:
    # 2 calls per step (pre-sweep, post-sweep) for 3 steps
    rates_step1 = {"FeS": np.array([1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    rates_step2 = {"FeS": np.array([2e-7, 2e-7, 2e-7]), "TS2": np.array([2e-7, 2e-7, 2e-7])}
    mock_update_coeffs.side_effect = [
        rates_step1, rates_step1,  # Step 1
        rates_step2, rates_step2,  # Step 2
        rates_step2, rates_step2   # Step 3
    ]
    
    # Call the solver
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    # The solver should successfully run up to max_steps (3 steps here)
    assert step == 3

@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_adaptation_sign_change_rollback(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that consecutive sign changes in rates (oscillation) triggers rollback and dt reduction."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    mp.max_steps = 4
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    # RATES sequence (consecutive sign flips required for rejection):
    # 1. Step 1 -> positive rate (2 calls) -> accepted
    # 2. Step 2 -> negative rate (2 calls) -> first flip -> accepted
    # 3. Step 3 tentative -> positive rate (2 calls) -> second consecutive flip -> rejected & rollback
    # 4. Step 3 retry -> negative rate (2 calls) -> accepted
    # 5. Step 4 -> negative rate (2 calls) -> accepted
    rates_pos = {"FeS": np.array([1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    rates_neg = {"FeS": np.array([-1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    
    mock_update_coeffs.side_effect = [
        rates_pos, rates_pos,          # Step 1
        rates_neg, rates_neg,          # Step 2
        rates_pos, rates_pos,          # Step 3 tentative -> rejected (consecutive flip)
        rates_neg, rates_neg,          # Step 3 retry -> accepted
        rates_neg, rates_neg           # Step 4
    ]
    
    # Track dt values to see if it cut
    dts = []
    def sweep_side_effect(dt, solver):
        dts.append(dt)
        return 0.0
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    # Verify rejection and rollback happened at Step 3
    # dts should contain:
    # 1. Step 1: 5.0 (dt_init)
    # 2. Step 2: 6.0 (5.0 * 1.2 growth)
    # 3. Step 3 tentative: 7.2 (6.0 * 1.2 growth)
    # 4. Step 3 retry: 3.6 (7.2 * 0.5 cut)
    # 5. Step 4: 4.32 (3.6 * 1.2 growth)
    assert len(dts) >= 5
    assert dts[3] == pytest.approx(3.6)

@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_adaptation_magnitude_rollback(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that a >1-order-of-magnitude change in rates triggers rollback and dt reduction."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    # RATES sequence:
    # 1. Step 1 -> accepted (2 calls)
    # 2. Step 2 tentative -> >10x increase -> rejected (2 calls)
    # 3. Step 2 retry -> accepted (2 calls)
    # 4. Step 3 -> accepted (2 calls)
    rates_init = {"FeS": np.array([1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    rates_huge_jump = {"FeS": np.array([2e-6, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    rates_retry = {"FeS": np.array([1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    
    mock_update_coeffs.side_effect = [
        rates_init, rates_init,
        rates_huge_jump, rates_huge_jump,
        rates_retry, rates_retry,
        rates_retry, rates_retry
    ]
    
    # Track dt values to see if it cut
    dts = []
    def sweep_side_effect(dt, solver):
        dts.append(dt)
        return 0.0
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert len(dts) >= 4
    assert dts[2] == pytest.approx(3.0)

@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_adaptation_noise_ignored(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that rate changes below the noise threshold are ignored."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    rates_init = {"FeS": np.array([1e-9, 1e-9, 1e-9]), "TS2": np.array([1e-9, 1e-9, 1e-9])}
    rates_noise_oscillating = {"FeS": np.array([-1e-9, 1e-9, 1e-9]), "TS2": np.array([1e-9, 1e-9, 1e-9])}
    
    mock_update_coeffs.side_effect = [
        rates_init, rates_init,
        rates_noise_oscillating, rates_noise_oscillating,
        rates_noise_oscillating, rates_noise_oscillating
    ]
    
    dts = []
    def sweep_side_effect(dt, solver):
        dts.append(dt)
        return 0.0
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert step == 3
    assert len(dts) == 3
    assert dts[1] == pytest.approx(6.0)
    assert dts[2] == pytest.approx(7.2)
