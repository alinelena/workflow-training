"""Trajectory and Structure Sampling for Active Learning.

Implements Farthest Point Sampling (FPS) in MLIP descriptor/representation space,
uniform striding, and diversity selection to avoid redundant reference DFT labeling.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from ase.atoms import Atoms

logger = logging.getLogger(__name__)


def extract_frame_descriptors(
    frames: list[Atoms],
    calculator: Any | None = None,
    arch: str = "mace_mp",
    model: str = "mh-1",
    head: str = "matpes_r2scan",
    device: str = "cpu",
) -> np.ndarray:
    """Extract invariant descriptor representations for each frame.

    Uses the model's native descriptor space (e.g. 1024-D MACE representations)
    averaged across atoms in each periodic or molecular configuration.
    """
    if not frames:
        return np.empty((0, 0), dtype=np.float32)

    # Initialize calculator if not provided
    if calculator is None:
        try:
            from mace.calculators import mace_mp

            calculator = mace_mp(
                model=model,
                head=head,
                device=device,
                default_dtype="float32",
            )
        except Exception as e:
            logger.warning(
                f"Could not load native MACE calculator for descriptors ({e}), "
                "falling back to structural coordinates."
            )
            calculator = None

    descriptors_list: list[np.ndarray] = []

    for i, atoms in enumerate(frames):
        desc_vec = None
        if calculator is not None and hasattr(calculator, "get_descriptors"):
            try:
                # Shape: (num_atoms, n_features)
                atom_descs = calculator.get_descriptors(atoms)
                if atom_descs is not None and len(atom_descs) > 0:
                    desc_vec = np.mean(atom_descs, axis=0)
            except Exception as e:
                logger.debug(f"Frame {i}: calculator.get_descriptors failed: {e}")

        if desc_vec is None:
            # Fallback representation: sorted pairwise atomic distance histogram or positions
            pos = atoms.get_positions()
            cell = atoms.get_cell().diagonal()
            desc_vec = np.concatenate([pos.mean(axis=0), pos.std(axis=0), cell])

        descriptors_list.append(np.array(desc_vec, dtype=np.float32).flatten())

    return np.vstack(descriptors_list)


def farthest_point_sampling(
    feature_matrix: np.ndarray,
    n_samples: int,
    initial_idx: int = 0,
) -> list[int]:
    """Perform Farthest Point Sampling (FPS) on a feature representation matrix.

    Iteratively selects the candidate with the maximum minimum Euclidean distance
    to all currently selected candidates.

    Args:
        feature_matrix: Shape (n_points, n_dimensions).
        n_samples: Number of diverse samples to select.
        initial_idx: Index of the initial seed point (default: 0).

    Returns:
        List of selected indices of length min(n_samples, n_points).
    """
    n_points = feature_matrix.shape[0]
    if n_samples >= n_points:
        return list(range(n_points))
    if n_samples <= 0:
        return []

    selected_indices: list[int] = [initial_idx]
    # Initialize distances to distance from the first selected point
    min_distances = np.linalg.norm(feature_matrix - feature_matrix[initial_idx], axis=1)

    for _ in range(1, n_samples):
        # Pick the candidate point farthest from the existing selected set
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)

        # Update min distance for all points
        new_dists = np.linalg.norm(feature_matrix - feature_matrix[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_dists)

    return selected_indices


def sample_trajectory_frames(
    frames: list[Atoms],
    n_samples: int,
    method: str = "fps",
    calculator: Any | None = None,
    random_seed: int = 42,
) -> list[Atoms]:
    """Sample diverse configurations from a list of frames.

    Supported methods:
        - "fps": Farthest Point Sampling in MLIP descriptor space.
        - "stride" / "uniform": Uniform temporal striding.
        - "random": Uniform random selection without replacement.
        - "all": Return all frames.

    Args:
        frames: Pool of candidate configurations.
        n_samples: Target number of configurations to extract.
        method: Selection strategy ("fps", "stride", "random", "all").
        calculator: Potential calculator used for extracting representations.
        random_seed: Random seed for stochastic sampling.

    Returns:
        List of selected ASE Atoms objects.
    """
    total_frames = len(frames)
    if total_frames <= n_samples or method == "all":
        return [f.copy() for f in frames]

    if method == "fps":
        feature_matrix = extract_frame_descriptors(frames, calculator=calculator)
        selected_indices = farthest_point_sampling(feature_matrix, n_samples=n_samples)
    elif method in ("stride", "uniform"):
        step = max(1, total_frames // n_samples)
        selected_indices = list(range(0, total_frames, step))[:n_samples]
    elif method == "random":
        rng = np.random.default_rng(random_seed)
        selected_indices = sorted(rng.choice(total_frames, size=n_samples, replace=False).tolist())
    else:
        raise ValueError(
            f"Unknown sampling method: {method}. Choose from 'fps', 'stride', 'random', 'all'."
        )

    sampled_frames = []
    for rank, idx in enumerate(selected_indices):
        f = frames[idx].copy()
        f.info["sample_method"] = method
        f.info["sample_rank"] = rank
        f.info["original_trajectory_index"] = idx
        sampled_frames.append(f)

    return sampled_frames
