"""
Tests for 5-Stage Invariance Gate and select_next_model guard.
"""

from ase import Atoms

from workflow_training.active_learning.invariance_gate import (
    check_eos_bulk_modulus,
    check_gamma_phonons,
    check_ground_state_forces,
    check_so3_equivariance,
    evaluate_invariance_gate,
    select_next_model,
)


def test_individual_physical_checks(silicon_structure: Atoms):
    # 1. Force check on relaxed structure
    p1, f_max, f_rms = check_ground_state_forces(silicon_structure, f_max_tol=0.025)
    assert p1 is True
    assert f_max <= 0.025

    # 2. EOS check
    p2, b0_gpa, delta_b0 = check_eos_bulk_modulus(
        silicon_structure, baseline_b0_gpa=160.0, b0_tol_percent=8.0
    )
    assert delta_b0 <= 8.0

    # 3. Gamma phonon stability
    p3, min_k = check_gamma_phonons(silicon_structure, omega2_tol=-1e-3)
    assert p3 is True
    assert min_k >= -1e-3

    # 4. SO(3) Equivariance
    p4, e_drift, f_err = check_so3_equivariance(silicon_structure)
    assert p4 is True
    assert e_drift < 1e-4


def test_evaluate_invariance_gate(silicon_structure: Atoms, mock_model):
    report = evaluate_invariance_gate(
        candidate_model=mock_model,
        baseline_structure=silicon_structure,
        baseline_b0=160.0,
    )
    res = report.get_dict() if hasattr(report, "get_dict") else report
    assert "passed" in res
    assert "residual_force" in res
    assert "delta_b0_percent" in res
    assert "phonons_stable" in res
    assert res["passed"] is True


def test_select_next_model_hot_swap_guard(mock_model):
    fallback_model = {"name": "model_v0"}
    candidate_model = {"name": "model_v1"}

    # Case A: Gate passed -> promote candidate
    next_model_a = select_next_model(
        candidate_model=candidate_model,
        fallback_model=fallback_model,
        gate_result={"passed": True},
    )
    assert next_model_a == candidate_model

    # Case B: Gate rejected -> retain fallback
    next_model_b = select_next_model(
        candidate_model=candidate_model,
        fallback_model=fallback_model,
        gate_result={"passed": False, "reason": "bulk_modulus_diverged"},
    )
    assert next_model_b == fallback_model
