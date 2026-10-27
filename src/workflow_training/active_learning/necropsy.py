"""
Post-mortem causal analysis and inflection cusp extraction for crashed trajectories.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from workflow_training.active_learning.ring_buffer import TrajectoryRingBuffer


def _verify_frame_safety(
    atoms: Atoms,
    atomic_numbers: np.ndarray,
    safe_ratio: float,
) -> tuple[bool, float]:
    """
    Check if an atomic configuration satisfies min r_ij >= safe_ratio * (R_i + R_j).
    """
    if len(atoms) <= 1:
        return True, 1.0

    cov_r = covalent_radii[atomic_numbers]
    cov_sum = cov_r[:, None] + cov_r[None, :]

    dist = atoms.get_all_distances(mic=any(atoms.get_pbc()))
    np.fill_diagonal(dist, np.inf)

    ratio_matrix = dist / (cov_sum + 1e-12)
    min_ratio = float(np.min(ratio_matrix))
    return (min_ratio >= safe_ratio), min_ratio


def find_inflection_cusp(
    buffer: TrajectoryRingBuffer,
    atomic_numbers: np.ndarray,
    safe_ratio: float = 0.85,
    baseline_sigma_mult: float = 3.5,
) -> dict:
    """
    Execute reverse baseline Z-score inflection search across the trajectory ring buffer.

    Strictly discards the terminal crashed frame (slot -1) to prevent corrupting DFT SCF loops.
    Scans backward to extract the physical pre-collapse cusp structure t*.

    Parameters
    ----------
    buffer : TrajectoryRingBuffer
        Circular trajectory buffer populated during the simulation.
    atomic_numbers : np.ndarray
        Array of atomic numbers Z of length n_atoms.
    safe_ratio : float, default 0.85
        Minimum interatomic separation as a fraction of sum of covalent radii.
    baseline_sigma_mult : float, default 3.5
        Z-score cutoff for detecting the onset of non-linear divergence.

    Returns
    -------
    dict
        {
            "cusp_atoms": Atoms,
            "approach_atoms": Atoms | None,
            "cusp_step": int,
            "cusp_max_force": float,
            "discarded_crash_step": int,
            "reason": str,
            "is_safe_for_dft": bool
        }
    """
    n_frames = len(buffer)
    if n_frames == 0:
        raise ValueError("Cannot extract inflection cusp from an empty TrajectoryRingBuffer.")

    chron_slots = buffer.get_chronological_slots()
    step_history = buffer.get_step_history()
    force_history = buffer.get_force_history()

    # The terminal frame is slot -1 (corpse frame)
    discarded_crash_step = int(step_history[-1])

    # Handle early trajectory crashes (t < 5)
    if n_frames < 5:
        # Fall back to earliest available frame (slot 0)
        fallback_slot = chron_slots[0]
        cusp_atoms = buffer.reconstruct_atoms(fallback_slot, atomic_numbers)
        is_safe, min_ratio = _verify_frame_safety(cusp_atoms, atomic_numbers, safe_ratio)
        return {
            "cusp_atoms": cusp_atoms,
            "approach_atoms": None,
            "cusp_step": int(step_history[0]),
            "cusp_max_force": float(force_history[0]),
            "discarded_crash_step": discarded_crash_step,
            "reason": f"early_crash_fallback(frames={n_frames}, min_ratio={min_ratio:.3f})",
            "is_safe_for_dft": is_safe,
        }

    # Baseline statistics from stable pre-crash window (exclude last 5 frames before crash)
    baseline_window = force_history[:-5]
    if len(baseline_window) == 0:
        baseline_window = force_history[: max(1, n_frames // 2)]

    median_f = float(np.median(baseline_window))
    # Median Absolute Deviation (MAD) scaled to normal distribution equivalent
    mad_f = float(1.4826 * np.median(np.abs(baseline_window - median_f)))
    sigma_f = max(mad_f, 1e-4)

    # Scan backward from slot -2 (frame right before terminal crash)
    selected_idx = None
    for i in range(n_frames - 2, -1, -1):
        f_val = force_history[i]
        z_score = (f_val - median_f) / sigma_f

        # Check if force is within normal statistical fluctuations
        if z_score <= baseline_sigma_mult:
            cand_slot = chron_slots[i]
            cand_atoms = buffer.reconstruct_atoms(cand_slot, atomic_numbers)
            is_safe, _ = _verify_frame_safety(cand_atoms, atomic_numbers, safe_ratio)
            if is_safe:
                selected_idx = i
                break

    # If no frame met both criteria, scan up to 10 frames back specifically for covalent safety
    if selected_idx is None:
        max_lookback = min(10, n_frames - 1)
        for i in range(n_frames - 2, n_frames - 2 - max_lookback, -1):
            if i < 0:
                break
            cand_slot = chron_slots[i]
            cand_atoms = buffer.reconstruct_atoms(cand_slot, atomic_numbers)
            is_safe, _ = _verify_frame_safety(cand_atoms, atomic_numbers, safe_ratio)
            if is_safe:
                selected_idx = i
                break

    # If still not found, select frame 0 as safest ground-state anchor
    if selected_idx is None:
        selected_idx = 0

    cusp_slot = chron_slots[selected_idx]
    cusp_atoms = buffer.reconstruct_atoms(cusp_slot, atomic_numbers)
    cusp_step = int(step_history[selected_idx])
    cusp_max_force = float(force_history[selected_idx])
    is_safe, min_ratio = _verify_frame_safety(cusp_atoms, atomic_numbers, safe_ratio)

    # Approach atoms (5 frames before cusp if available)
    approach_atoms = None
    if selected_idx >= 5:
        app_slot = chron_slots[selected_idx - 5]
        approach_atoms = buffer.reconstruct_atoms(app_slot, atomic_numbers)

    return {
        "cusp_atoms": cusp_atoms,
        "approach_atoms": approach_atoms,
        "cusp_step": cusp_step,
        "cusp_max_force": cusp_max_force,
        "discarded_crash_step": discarded_crash_step,
        "reason": f"reverse_z_score(idx={selected_idx}, z_max={baseline_sigma_mult}, min_ratio={min_ratio:.3f})",
        "is_safe_for_dft": is_safe,
    }
