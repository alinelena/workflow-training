"""
Tests for Post-Mortem Necropsy Engine: reverse Z-score inflection search, corpse rejection, and early crash handling.
"""

import numpy as np
from ase import Atoms

from workflow_training.active_learning.necropsy import find_inflection_cusp
from workflow_training.active_learning.ring_buffer import TrajectoryRingBuffer


def test_corpse_frame_strictly_discarded(silicon_structure: Atoms):
    n_atoms = len(silicon_structure)
    buffer = TrajectoryRingBuffer(capacity=20, n_atoms=n_atoms)
    numbers = silicon_structure.get_atomic_numbers()

    # Populate 15 stable steps with small random forces (~0.5 eV/Å)
    for step in range(15):
        f = np.random.normal(0.0, 0.3, size=(n_atoms, 3))
        buffer.append(step, silicon_structure, f, -10.0, 0.5)

    # Step 15: Cusp step where force begins diverging to 2.5 eV/Å (safe)
    f_cusp = np.full((n_atoms, 3), 1.4)
    buffer.append(15, silicon_structure, f_cusp, -9.5, 0.8)

    # Step 16: Terminal corpse frame (catastrophic crash with 35.0 eV/Å and atom clash)
    clashed = silicon_structure.copy()
    pos = clashed.get_positions()
    pos[1] = pos[0] + [0.4, 0.0, 0.0]  # Severe clash
    clashed.set_positions(pos)
    f_crash = np.full((n_atoms, 3), 20.0)  # > 34 eV/Å norm
    buffer.append(16, clashed, f_crash, 100.0, 50.0)

    result = find_inflection_cusp(buffer, numbers, safe_ratio=0.85, baseline_sigma_mult=3.5)

    # The corpse frame (step 16) MUST be recorded as discarded
    assert result["discarded_crash_step"] == 16

    # The extracted cusp frame must NOT be step 16
    assert result["cusp_step"] != 16
    assert result["is_safe_for_dft"] is True

    # Reconstructed cusp must satisfy covalent safety
    cusp_atoms = result["cusp_atoms"]
    dists = cusp_atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    assert np.min(dists) > 1.5  # Safe distance


def test_early_trajectory_crash_handling(silicon_structure: Atoms):
    n_atoms = len(silicon_structure)
    buffer = TrajectoryRingBuffer(capacity=10, n_atoms=n_atoms)
    numbers = silicon_structure.get_atomic_numbers()

    # Model trips on step 2 (buffer has only 3 frames: steps 0, 1, 2)
    buffer.append(0, silicon_structure, np.zeros((n_atoms, 3)), -10.0, 0.1)
    buffer.append(1, silicon_structure, np.ones((n_atoms, 3)), -9.8, 0.2)

    # Crash frame
    buffer.append(2, silicon_structure, np.full((n_atoms, 3), 30.0), 50.0, 10.0)

    assert len(buffer) == 3

    # Should not raise; falls back gracefully to step 0 reference configuration
    result = find_inflection_cusp(buffer, numbers, safe_ratio=0.85)

    assert result["discarded_crash_step"] == 2
    assert result["cusp_step"] == 0
    assert result["is_safe_for_dft"] is True
    assert "early_crash_fallback" in result["reason"]
