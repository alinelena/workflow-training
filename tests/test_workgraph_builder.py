"""
Tests for WorkGraph builder: structure validation, context schema, and concurrency limits.
"""

import numpy as np
import pytest
from ase import Atoms

from workflow_training.active_learning.workgraph import (
    build_guardrailed_al_workgraph,
    evaluate_loop_condition,
    increment_epoch_and_update_state,
    validate_structure,
)


def test_structure_validation_pass(silicon_structure: Atoms):
    assert validate_structure(silicon_structure) is True


def test_structure_validation_fail_clash():
    clashed = Atoms("Si2", positions=[[0, 0, 0], [0.3, 0.0, 0.0]], cell=5.43 * np.eye(3), pbc=True)
    with pytest.raises(ValueError, match="unphysical atomic overlap"):
        validate_structure(clashed)


def test_build_guardrailed_al_workgraph(
    silicon_structure: Atoms,
    mock_model,
    mock_pw_code,
    mock_janus_code,
    curriculum_config: dict,
):
    wg_info = build_guardrailed_al_workgraph(
        structure=silicon_structure,
        model=mock_model,
        pw_code=mock_pw_code,
        janus_code=mock_janus_code,
        pseudo_family="SSSP",
        curriculum_config=curriculum_config,
        max_number_jobs=16,
    )

    assert wg_info is not None
    if isinstance(wg_info, dict):
        assert wg_info["max_number_jobs"] == 16
        assert "current_model" in wg_info["context_keys"]
        assert "epoch_idx" in wg_info["context_keys"]
        assert "is_converged" in wg_info["context_keys"]


def test_evaluate_loop_condition():
    # Continues if not converged and epoch < max_epochs
    assert bool(evaluate_loop_condition(epoch=0, is_converged=False, max_epochs=5)) is True
    assert bool(evaluate_loop_condition(epoch=4, is_converged=False, max_epochs=5)) is True

    # Terminates if converged
    assert bool(evaluate_loop_condition(epoch=2, is_converged=True, max_epochs=5)) is False

    # Terminates if max_epochs reached
    assert bool(evaluate_loop_condition(epoch=5, is_converged=False, max_epochs=5)) is False


def test_increment_epoch_and_update_state():
    # Pass gate and advance temperature
    state1 = increment_epoch_and_update_state(
        current_epoch=0,
        gate_result={"passed": True},
        current_temp_idx=0,
        n_temps=3,
    )
    assert int(state1["next_epoch"]) == 1
    assert int(state1["next_temp_idx"]) == 1
    assert bool(state1["is_converged"]) is False

    # Pass gate at highest temperature -> convergence
    state2 = increment_epoch_and_update_state(
        current_epoch=2,
        gate_result={"passed": True},
        current_temp_idx=2,
        n_temps=3,
    )
    assert bool(state2["is_converged"]) is True

    # Gate rejected -> do not advance temperature, not converged
    state3 = increment_epoch_and_update_state(
        current_epoch=1,
        gate_result={"passed": False},
        current_temp_idx=1,
        n_temps=3,
    )
    assert int(state3["next_epoch"]) == 2
    assert int(state3["next_temp_idx"]) == 1
    assert bool(state3["is_converged"]) is False
