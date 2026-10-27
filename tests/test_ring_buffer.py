"""
Tests for TrajectoryRingBuffer: capacity, append latency (< 15 µs), and chronological recovery.
"""

import time

import numpy as np
from ase import Atoms

from workflow_training.active_learning.ring_buffer import TrajectoryRingBuffer


def test_buffer_initialization():
    buf = TrajectoryRingBuffer(capacity=50, n_atoms=4)
    assert len(buf) == 0
    assert buf.capacity == 50
    assert buf.n_atoms == 4


def test_buffer_append_and_latency(silicon_structure: Atoms):
    n_atoms = len(silicon_structure)
    capacity = 100
    buf = TrajectoryRingBuffer(capacity=capacity, n_atoms=n_atoms)

    dummy_forces = np.ones((n_atoms, 3), dtype=np.float64)

    # Measure per-step append latency
    start = time.perf_counter()
    n_iterations = 1000
    for step in range(n_iterations):
        buf.append(step, silicon_structure, dummy_forces, epot=-10.5, ekin=1.2)
    elapsed = time.perf_counter() - start

    avg_latency_us = (elapsed / n_iterations) * 1e6
    assert len(buf) == capacity
    # Latency budget: strictly < 15 microseconds
    assert avg_latency_us < 15.0, f"Average latency {avg_latency_us:.2f} µs exceeded 15 µs"


def test_buffer_chronological_wrapping(silicon_structure: Atoms):
    n_atoms = len(silicon_structure)
    capacity = 5
    buf = TrajectoryRingBuffer(capacity=capacity, n_atoms=n_atoms)

    for i in range(8):
        forces = np.full((n_atoms, 3), float(i))
        buf.append(step=i, atoms=silicon_structure, forces=forces, epot=float(i), ekin=0.5)

    assert len(buf) == 5
    steps = buf.get_step_history()
    # Oldest to newest should be 3, 4, 5, 6, 7
    np.testing.assert_array_equal(steps, np.array([3, 4, 5, 6, 7]))

    f_hist = buf.get_force_history()
    # Norm of [i, i, i] is i * sqrt(3)
    expected_f = np.array([3, 4, 5, 6, 7]) * np.sqrt(3)
    np.testing.assert_allclose(f_hist, expected_f)


def test_reconstruct_atoms(silicon_structure: Atoms):
    n_atoms = len(silicon_structure)
    buf = TrajectoryRingBuffer(capacity=10, n_atoms=n_atoms)

    silicon_structure.set_velocities(np.ones((n_atoms, 3)))
    buf.append(0, silicon_structure, np.zeros((n_atoms, 3)), -5.0, 0.5)

    reconstructed = buf.reconstruct_atoms(0, silicon_structure.get_atomic_numbers())
    assert len(reconstructed) == n_atoms
    np.testing.assert_allclose(reconstructed.get_positions(), silicon_structure.get_positions())
    np.testing.assert_allclose(reconstructed.get_velocities(), np.ones((n_atoms, 3)))
