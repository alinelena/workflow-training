"""
DFT output auditing, quarantine isolation, and 20% anchor replay dataset accumulation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read as ase_read
from ase.io import write as ase_write

try:
    from aiida import orm
    from aiida_workgraph import task
except ImportError:
    orm = None
    task = None


def _calc_or_func(func):
    if task is not None and hasattr(task, "calcfunction"):
        try:
            func.task = task.calcfunction(func)
        except Exception:
            func.task = None
    return func


def check_covalent_safety(atoms: Atoms, safe_ratio: float = 0.85) -> tuple[bool, float]:
    """Verify that all atomic pairs satisfy r_ij >= safe_ratio * (R_i + R_j)."""
    if len(atoms) <= 1:
        return True, 1.0

    numbers = atoms.get_atomic_numbers()
    cov_r = covalent_radii[numbers]
    cov_sum = cov_r[:, None] + cov_r[None, :]

    dist = atoms.get_all_distances(mic=any(atoms.get_pbc()))
    np.fill_diagonal(dist, np.inf)

    ratio_matrix = dist / (cov_sum + 1e-12)
    min_ratio = float(np.min(ratio_matrix))
    return (min_ratio >= safe_ratio), min_ratio


@_calc_or_func
def collect_filter_and_accumulate_dataset(
    previous_dataset: orm.SinglefileData if orm else object,
    anchor_group_label: orm.Str if orm else str,
    epoch_idx: orm.Int if orm else int,
    anchor_replay_ratio: orm.Float if orm else float = 0.20,
    **gathered_dft,
) -> dict:
    """
    Audit DFT results, quarantine electronic failures, and blend 80% new with 20% anchor replay.

    Accepts dynamic keyword arguments (**gathered_dft) from mapped DFT tasks to maintain
    unbroken provenance tracking in the AiiDA graph.
    """
    valid_frames: list[Atoms] = []
    quarantined_runs: list[dict] = []

    # 1. Parse and audit all incoming DFT candidate calculations
    for key, dft_output in gathered_dft.items():
        if hasattr(dft_output, "get_dict"):
            data = dft_output.get_dict()
        elif isinstance(dft_output, dict):
            data = dft_output
        elif isinstance(dft_output, Atoms):
            data = {
                "scf_converged": True,
                "energy": dft_output.info.get("energy", 0.0),
                "forces": dft_output.arrays.get("forces", None),
                "structure": dft_output,
            }
        else:
            try:
                data = dict(dft_output)
            except Exception:
                data = {"raw": dft_output}
        scf_converged = data.get("scf_converged", True)
        energy = data.get("energy", None)
        forces = data.get("forces", None)
        structure_dict = data.get("structure", None)

        if not scf_converged or energy is None:
            quarantined_runs.append(
                {
                    "source": key,
                    "reason": "scf_failed_or_missing_energy",
                    "details": data.get("error_message", "Non-convergent electronic loop"),
                }
            )
            continue

        # Reconstruct Atoms
        if structure_dict is not None and "positions" in structure_dict:
            atoms = Atoms(
                numbers=structure_dict.get("numbers", [1] * len(structure_dict["positions"])),
                positions=structure_dict["positions"],
                cell=structure_dict.get("cell", np.eye(3)),
                pbc=structure_dict.get("pbc", [True, True, True]),
            )
        else:
            atoms = None

        if atoms is not None:
            is_safe, min_ratio = check_covalent_safety(atoms, safe_ratio=0.85)
            if not is_safe:
                quarantined_runs.append(
                    {
                        "source": key,
                        "reason": "sub_covalent_clash",
                        "min_ratio": min_ratio,
                        "threshold": 0.85,
                    }
                )
                continue

            atoms.info["energy"] = float(energy)
            if forces is not None:
                atoms.arrays["forces"] = np.array(forces, dtype=np.float64)
            valid_frames.append(atoms)

    # 2. Read existing cumulative dataset
    existing_atoms: list[Atoms] = []
    if previous_dataset is not None:
        try:
            if hasattr(previous_dataset, "get_content"):
                with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as f:
                    f.write(previous_dataset.get_content())
                    f_name = f.name
                existing_atoms = ase_read(f_name, index=":")
                Path(f_name).unlink(missing_ok=True)
            elif isinstance(previous_dataset, (str, Path)) and Path(previous_dataset).exists():
                existing_atoms = ase_read(str(previous_dataset), index=":")
        except Exception:
            existing_atoms = []

    n_valid = len(valid_frames)
    n_quarantined = len(quarantined_runs)

    # 3. Handle 100% electronic quarantine (no new valid structures)
    if n_valid == 0:
        quarantine_report = {
            "epoch": int(epoch_idx.value) if hasattr(epoch_idx, "value") else int(epoch_idx),
            "n_valid": 0,
            "n_quarantined": n_quarantined,
            "quarantined_runs": quarantined_runs,
            "note": "100% quarantine or no candidates; retaining previous dataset unchanged.",
        }
        return {
            "updated_dataset": previous_dataset,
            "quarantine_report": orm.Dict(dict=quarantine_report) if orm else quarantine_report,
            "n_valid": orm.Int(0) if orm else 0,
            "n_quarantined": orm.Int(n_quarantined) if orm else n_quarantined,
        }

    # 4. Blend valid frames with 20% anchor replay from Stage 0
    anchor_frames: list[Atoms] = []
    label_str = (
        anchor_group_label.value
        if hasattr(anchor_group_label, "value")
        else str(anchor_group_label)
    )
    if orm is not None:
        try:
            anchor_group = orm.load_group(label_str)
            for node in anchor_group.nodes:
                if hasattr(node, "get_ase"):
                    anchor_frames.append(node.get_ase())
        except Exception:
            pass

    # If anchor frames unavailable from group, reuse earliest existing frames
    if len(anchor_frames) == 0 and len(existing_atoms) > 0:
        anchor_frames = existing_atoms[: min(10, len(existing_atoms))]

    rep_ratio = (
        float(anchor_replay_ratio.value)
        if hasattr(anchor_replay_ratio, "value")
        else float(anchor_replay_ratio)
    )
    n_anchors = (
        max(1, int(n_valid * rep_ratio / (1.0 - rep_ratio))) if len(anchor_frames) > 0 else 0
    )

    sampled_anchors: list[Atoms] = []
    if n_anchors > 0 and len(anchor_frames) > 0:
        indices = np.random.choice(
            len(anchor_frames), size=min(n_anchors, len(anchor_frames)), replace=False
        )
        sampled_anchors = [anchor_frames[i] for i in indices]

    # Combine existing + new valid + replay
    combined_dataset = existing_atoms + valid_frames + sampled_anchors

    # 5. Write updated .xyz
    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as f:
        ase_write(f.name, combined_dataset, format="extxyz")
        xyz_path = f.name

    can_use_orm = False
    if orm is not None:
        try:
            from aiida.manage.configuration import get_profile

            can_use_orm = get_profile() is not None
        except Exception:
            can_use_orm = False

    try:
        updated_dataset = orm.SinglefileData(file=xyz_path) if can_use_orm else xyz_path
    finally:
        if can_use_orm:
            Path(xyz_path).unlink(missing_ok=True)

    quarantine_report = {
        "epoch": int(epoch_idx.value) if hasattr(epoch_idx, "value") else int(epoch_idx),
        "n_valid": n_valid,
        "n_quarantined": n_quarantined,
        "n_anchors_replayed": len(sampled_anchors),
        "total_dataset_size": len(combined_dataset),
        "quarantined_runs": quarantined_runs,
    }

    if can_use_orm:
        return {
            "updated_dataset": updated_dataset,
            "quarantine_report": orm.Dict(dict=quarantine_report),
            "n_valid": orm.Int(n_valid),
            "n_quarantined": orm.Int(n_quarantined),
        }
    return {
        "updated_dataset": updated_dataset,
        "quarantine_report": quarantine_report,
        "n_valid": n_valid,
        "n_quarantined": n_quarantined,
    }
