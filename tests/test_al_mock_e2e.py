"""
End-to-end mock integration test verifying complete curriculum active learning cycle:
Stage 0 -> Guarded MD -> Quarantine -> Retrain Config -> Invariance Gate -> Model Guard.
"""

import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from workflow_training.active_learning.dataset_collector import (
    collect_filter_and_accumulate_dataset,
)
from workflow_training.active_learning.invariance_gate import (
    evaluate_invariance_gate,
    select_next_model,
)
from workflow_training.active_learning.stages.stage0_cold_start import (
    generate_cold_start_structures,
)
from workflow_training.active_learning.stages.stage1_thermal import run_guarded_md_task
from workflow_training.active_learning.training_config import prepare_mace_training_config


def test_al_full_cycle_mock_e2e(silicon_structure: Atoms, curriculum_config: dict, mock_model):
    # -------------------------------------------------------------
    # 1. Stage 0: Cold-start structure generation
    # -------------------------------------------------------------
    stage0_structures = generate_cold_start_structures(
        base_atoms=silicon_structure,
        strain_range=[-0.05, 0.0, 0.05],
        n_rattles=2,
    )
    assert len(stage0_structures) == 5

    # Write initial dataset file
    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as f:
        ase_write(f.name, stage0_structures, format="extxyz")
        init_dataset_path = f.name

    try:
        # -------------------------------------------------------------
        # 2. Stage 1: Guarded MD with Tri-Zone Candidate Selection
        # -------------------------------------------------------------
        md_result = run_guarded_md_task(
            structure=silicon_structure,
            model=mock_model,
            temperature=300.0,
            pressure_bar=0.0,
            md_steps=10,
            timestep_fs=1.0,
            tri_zone_config=curriculum_config["tri_zone"],
            necropsy_config=curriculum_config["necropsy"],
        )
        assert md_result["status"] == "completed"

        # -------------------------------------------------------------
        # 3. Mapped DFT Jobs & Quarantine Routing
        # -------------------------------------------------------------
        # Mock DFT results:
        # - calc_0: Converged normal DFT calculation
        # - calc_1: SCF convergence failure (should be quarantined)
        # - calc_2: Severe atomic overlap (should be quarantined)
        mock_dft_outputs = {
            "calc_0": {
                "scf_converged": True,
                "energy": -15.82,
                "forces": [[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]],
                "structure": {
                    "numbers": [14, 14],
                    "positions": silicon_structure.get_positions().tolist(),
                    "cell": silicon_structure.get_cell().tolist(),
                    "pbc": [True, True, True],
                },
            },
            "calc_1": {
                "scf_converged": False,
                "energy": None,
                "error_message": "Davidson loop failed to converge in 100 iterations",
            },
            "calc_2": {
                "scf_converged": True,
                "energy": -12.0,
                "structure": {
                    "numbers": [14, 14],
                    "positions": [[0.0, 0.0, 0.0], [0.35, 0.0, 0.0]],  # Severe clash
                    "cell": silicon_structure.get_cell().tolist(),
                    "pbc": [True, True, True],
                },
            },
        }

        collector_result = collect_filter_and_accumulate_dataset(
            previous_dataset=init_dataset_path,
            anchor_group_label="stage0_anchors",
            epoch_idx=1,
            anchor_replay_ratio=0.20,
            **mock_dft_outputs,
        )

        assert collector_result["n_valid"] == 1
        assert collector_result["n_quarantined"] == 2
        report = (
            collector_result["quarantine_report"].get_dict()
            if hasattr(collector_result["quarantine_report"], "get_dict")
            else collector_result["quarantine_report"]
        )
        assert report["n_quarantined"] == 2

        # -------------------------------------------------------------
        # 4. MACE Retraining Config Preparation
        # -------------------------------------------------------------
        train_config = prepare_mace_training_config(
            dataset_file=collector_result["updated_dataset"],
            base_model=mock_model,
            training_params=curriculum_config["training"],
        )
        assert train_config is not None

        # -------------------------------------------------------------
        # 5. 5-Stage Invariance Gate Evaluation
        # -------------------------------------------------------------
        candidate_model = {"model_name": "mace_finetuned_epoch_1"}
        gate_res = evaluate_invariance_gate(
            candidate_model=candidate_model,
            baseline_structure=silicon_structure,
            baseline_b0=160.0,
        )
        gate_dict = gate_res.get_dict() if hasattr(gate_res, "get_dict") else gate_res
        assert gate_dict["passed"] is True

        # -------------------------------------------------------------
        # 6. Model Selector Hot-Swap Guard
        # -------------------------------------------------------------
        promoted_model = select_next_model(
            candidate_model=candidate_model,
            fallback_model=mock_model,
            gate_result=gate_res,
        )
        assert promoted_model == candidate_model

    finally:
        Path(init_dataset_path).unlink(missing_ok=True)
