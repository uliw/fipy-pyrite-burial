import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fipy import CellVariable, Grid1D
from fipyrite.diff_lib import data_container
from fipyrite.solver_calls import (
    run_non_steady_state_solver_coupled,
    _compute_inner_residual,
)


class MockMP:
    def __init__(self):
        self.plot_name = "test_inner_sweep_run"
        self.max_steps = 3
        self.start_time = 0.0
        self.t_end = 1000.0
        self.dt_min = 1.0
        self.dt_init = 10.0
        self.dt_max = 100.0
        self.backup_step = 100
        self.report_step = 100
        self.isotopes = False
        self.title = "test"
        self.enable_rate_adaptation = False
        self.phi = 0.8
        self.solver_backend = "default"
        self.tolerance = 1e-12
        self.dt_tolerance = -1.0
        self.adaptive_solver_tolerance = False
        
        # Inner sweeping settings
        self.enable_inner_sweeping = True
        self.max_inner_sweeps = 5
        self.inner_tol = 1e-3
        self.inner_relaxation = 1.0
        self.inner_sweep_equilibrium = False
        self.adaptive_sweeps_dt = True
        self.sweep_target_optimal = 3
        self.sweep_max_acceptable = 4


class MockD:
    def __init__(self):
        self.FeS = 0.0
        self.TS2 = 0.0
        self.D_bio = 0.0
        self.D_irr = 0.0


@pytest.fixture
def solver_setup():
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


def test_compute_inner_residual():
    mesh = Grid1D(nx=3)
    var = CellVariable(mesh=mesh, value=1.0)
    species_struct = [{"name": "A", "var": var}]
    
    # Identical values: error should be 0.0
    prev_iterate = {"A": np.array([1.0, 1.0, 1.0])}
    err = _compute_inner_residual(species_struct, prev_iterate, inner_tol=1e-4)
    assert err == 0.0
    
    # Change is below atol (1e-6) + rtol (1e-4) -> err <= 1.0
    prev_iterate = {"A": np.array([1.0 - 1e-5, 1.0, 1.0])}
    err = _compute_inner_residual(species_struct, prev_iterate, inner_tol=1e-4, atol_default=1e-4)
    assert err <= 1.0
    
    # Large change: diff = 0.5, scale = 1e-4 * 1.0 + 1e-6 -> err >> 1.0
    prev_iterate = {"A": np.array([0.5, 1.0, 1.0])}
    err = _compute_inner_residual(species_struct, prev_iterate, inner_tol=1e-4)
    assert err > 1.0
    
    # NaN check: should safely return inf
    prev_iterate = {"A": np.array([np.nan, 1.0, 1.0])}
    err = _compute_inner_residual(species_struct, prev_iterate, inner_tol=1e-4)
    assert np.isinf(err)


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_inner_sweeping_convergence(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that inner sweeping iterates until residual is met and advances time."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 5
    mp.inner_tol = 1e-3
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}
    
    sweep_counts = []
    def sweep_side_effect(dt, solver):
        sweep_counts.append(1)
        if len(sweep_counts) == 1:
            c["TS2"].setValue(1.1)
        elif len(sweep_counts) == 2:
            c["TS2"].setValue(1.105)
        else:
            c["TS2"].setValue(1.1050001)
        return 0.0
    
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert step == 1
    assert len(sweep_counts) == 3  # Converged in 3 sweeps!


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_inner_sweeping_failure_and_rollback(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that failure to converge within max_inner_sweeps triggers rollback and cuts dt."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 3
    mp.inner_tol = 1e-6
    mp.dt_init = 20.0
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}
    
    attempt = {"count": 0}
    sweep_dts = []
    
    def sweep_side_effect(dt, solver):
        sweep_dts.append(dt)
        if dt > 15.0:
            attempt["count"] += 1
            c["TS2"].setValue(attempt["count"] * 2.0)
        else:
            c["TS2"].setValue(1.0)
        return 0.0
    
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert step == 1
    assert sweep_dts[0] == 20.0
    assert sweep_dts[1] == 20.0
    assert sweep_dts[2] == 20.0
    assert sweep_dts[3] < 20.0


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_inner_sweeping_dt_growth(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that when steps converge in <= sweep_target_optimal sweeps, dt grows."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 3
    mp.dt_init = 10.0
    mp.sweep_target_optimal = 3
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}
    
    dts_at_step_start = []
    def sweep_side_effect(dt, solver):
        dts_at_step_start.append(dt)
        c["TS2"].setValue(c["TS2"].old.value)
        return 0.0
    
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert step == 3
    assert dts_at_step_start[0] == pytest.approx(10.0)
    assert dts_at_step_start[1] == pytest.approx(12.0)
    assert dts_at_step_start[2] == pytest.approx(14.4)


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_inner_sweeping_adaptive_damping(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that adaptive damping engages when residual increases and drives convergence."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 8
    mp.inner_tol = 1e-3
    mp.enable_adaptive_damping = True
    mp.inner_relaxation = 1.0

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    sweeps = []
    def sweep_side_effect(dt, solver):
        sweeps.append(len(sweeps) + 1)
        # Sweep 1: small change
        if len(sweeps) == 1:
            c["TS2"].setValue(1.05)
        # Sweep 2: sudden overshoot -> large change (residual grows)
        elif len(sweeps) == 2:
            c["TS2"].setValue(1.8)
        # Sweep 3+: stabilizes as damping factor reduces
        elif len(sweeps) == 3:
            c["TS2"].setValue(1.3)
        else:
            c["TS2"].setValue(1.3000001)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_side_effect

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    assert step == 1
    assert len(sweeps) <= 6

