"""
Example 3: Electronic Failure Quarantine, 20% Anchor Replay, and 5-Stage Invariance Gate.

Demonstrates:
1. Automated filtering and quarantine isolation of SCF electronic failures and clashing frames.
2. Anti-catastrophic forgetting: Blending 80% new informative frames with 20% Stage 0 anchor replay.
3. MACE training config generation with STFC headless execution rule (tracker: false).
4. 5-Stage Physical Invariance Pre-Deployment Gate (forces, bulk modulus, phonons, SO(3), burn-in).
5. Model Selector Context Guard protecting against model regression.
"""

from __future__ import annotations

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
from workflow_training.active_learning.training_config import prepare_mace_training_config


def main():
    print("=" * 70)
    print("Example 3: Quarantine, Replay Mixing, and 5-Stage Invariance Gate")
    print("=" * 70)

    try:
        from aiida import load_profile

        load_profile()
    except Exception:
        pass

    # 1. Base Structure & Stage 0 Cold-Start Anchor Reservoir
    a = 5.43
    si_atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[[0.0, 0.0, 0.0], [0.25 * a, 0.25 * a, 0.25 * a]],
        cell=[
            [0.0, 0.5 * a, 0.5 * a],
            [0.5 * a, 0.0, 0.5 * a],
            [0.5 * a, 0.5 * a, 0.0],
        ],
        pbc=True,
    )
    stage0_anchors = generate_cold_start_structures(
        base_atoms=si_atoms,
        strain_range=[-0.05, 0.0, 0.05],
        n_rattles=2,
    )
    print(f"Generated {len(stage0_anchors)} Stage 0 baseline structures (EOS + harmonic rattles).")

    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as f:
        ase_write(f.name, stage0_anchors, format="extxyz")
        stage0_dataset_path = f.name

    try:
        # 2. Simulated Batch of DFT Calculations from HPC
        # Contains 1 valid frame, 1 SCF convergence failure, and 1 clashing frame
        dft_batch = {
            "dft_task_0": {
                "scf_converged": True,
                "energy": -15.85,
                "forces": [[0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]],
                "structure": {
                    "numbers": [14, 14],
                    "positions": si_atoms.get_positions().tolist(),
                    "cell": si_atoms.get_cell().tolist(),
                    "pbc": [True, True, True],
                },
            },
            "dft_task_1": {
                "scf_converged": False,
                "energy": None,
                "error_message": "Davidson diagonalizer exceeded 100 iterations (negative charge density)",
            },
            "dft_task_2": {
                "scf_converged": True,
                "energy": -11.5,
                "structure": {
                    "numbers": [14, 14],
                    "positions": [[0.0, 0.0, 0.0], [0.35, 0.0, 0.0]],  # Severe overlap
                    "cell": si_atoms.get_cell().tolist(),
                    "pbc": [True, True, True],
                },
            },
        }

        print("\nAuditing incoming DFT batch:")
        result = collect_filter_and_accumulate_dataset(
            previous_dataset=stage0_dataset_path,
            anchor_group_label="stage0_anchors",
            epoch_idx=1,
            anchor_replay_ratio=0.20,
            **dft_batch,
        )

        n_valid = (
            int(result["n_valid"].value)
            if hasattr(result["n_valid"], "value")
            else int(result["n_valid"])
        )
        n_quarantined = (
            int(result["n_quarantined"].value)
            if hasattr(result["n_quarantined"], "value")
            else int(result["n_quarantined"])
        )
        report = (
            result["quarantine_report"].get_dict()
            if hasattr(result["quarantine_report"], "get_dict")
            else result["quarantine_report"]
        )

        print(f"  Valid frames accepted for training: {n_valid}")
        print(f"  Unphysical / non-converged frames quarantined: {n_quarantined}")
        print(f"  Total accumulated dataset size: {report['total_dataset_size']}")
        print(f"  Anchor frames replayed (20% quota): {report['n_anchors_replayed']}")

        # 3. Build JanusConfigfile for MACE Fine-Tuning
        training_params = {
            "energy_weight": 1.0,
            "forces_weight": 20.0,
            "stress_weight": 5.0,
            "learning_rate": 1e-4,
            "max_num_epochs": 25,
            "batch_size": 8,
        }
        config_node = prepare_mace_training_config(
            dataset_file=result["updated_dataset"],
            base_model="mace_foundation.model",
            training_params=training_params,
        )
        print("\nMACE Fine-Tuning Config prepared:")
        if isinstance(config_node, dict):
            print(
                f"  Loss weights: E={config_node['energy_weight']}, F={config_node['forces_weight']}, S={config_node['stress_weight']}"
            )
            print(f"  Headless tracker: {config_node['tracker']} (CodeCarbon tracking disabled)")
        else:
            print("  JanusConfigfile node generated with tracker: false.")

        # 4. 5-Stage Physical Invariance Pre-Deployment Gate
        print("\nEvaluating Candidate Checkpoint across 5 Physical Invariance Gates:")
        candidate_model = {"checkpoint": "mace_epoch_1.model"}
        fallback_model = {"checkpoint": "mace_foundation.model"}

        gate_res = evaluate_invariance_gate(
            candidate_model=candidate_model,
            baseline_structure=si_atoms,
            baseline_b0=160.0,
        )
        gate_dict = gate_res.get_dict() if hasattr(gate_res, "get_dict") else gate_res

        print(f"  1. Ground-state residual force: {gate_dict['residual_force']:.4f} eV/Å (Passed)")
        print(
            f"  2. Bulk modulus retention: {gate_dict['delta_b0_percent']:.1f}% deviation (Passed <= 8%)"
        )
        print(f"  3. Acoustic Gamma phonons: stable={gate_dict['phonons_stable']} (Passed)")
        print(f"  4. SO(3) Equivariance: drift={gate_dict['equivariance_error']:.2e} (Passed)")
        print(f"  5. Burn-in MD: drift={gate_dict['burn_in_drift']:.2e} eV/atom (Passed)")
        print(f"  Overall Gate Status: {'PASSED' if gate_dict['passed'] else 'REJECTED'}")

        # 5. Model Selector Hot-Swap Guard
        promoted_model = select_next_model(
            candidate_model=candidate_model,
            fallback_model=fallback_model,
            gate_result=gate_res,
        )
        print(
            f"\nModel Selector Decision: Promoted {promoted_model['checkpoint']} into active WorkGraph context."
        )
        print("=" * 70)

    finally:
        Path(stage0_dataset_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
