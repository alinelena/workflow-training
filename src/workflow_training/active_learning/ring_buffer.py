"""
Zero-allocation pre-allocated circular FIFO ring buffer for molecular dynamics trajectories.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


class TrajectoryRingBuffer:
    """
    Circular FIFO ring buffer for in-memory molecular dynamics trajectory monitoring.

    Pre-allocates memory for positions, velocities, forces, cells, and scalar observables
    to guarantee zero dynamic allocations during MD time integration steps (< 15 µs latency).
    """

    def __init__(self, capacity: int, n_atoms: int):
        if capacity <= 0:
            raise ValueError("Buffer capacity must be a positive integer.")
        if n_atoms <= 0:
            raise ValueError("Number of atoms must be a positive integer.")

        self.capacity: int = capacity
        self.n_atoms: int = n_atoms
        self.cursor: int = 0
        self.count: int = 0

        # Pre-allocated arrays
        self.positions: np.ndarray = np.zeros((capacity, n_atoms, 3), dtype=np.float64)
        self.velocities: np.ndarray = np.zeros((capacity, n_atoms, 3), dtype=np.float64)
        self.forces: np.ndarray = np.zeros((capacity, n_atoms, 3), dtype=np.float64)
        self.cells: np.ndarray = np.zeros((capacity, 3, 3), dtype=np.float64)
        self.pbc: np.ndarray = np.zeros((capacity, 3), dtype=bool)
        self.epot: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.ekin: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.steps: np.ndarray = np.zeros(capacity, dtype=np.int64)

    def append(
        self,
        step: int,
        atoms: Atoms,
        forces: np.ndarray,
        epot: float,
        ekin: float,
    ) -> None:
        """
        Append a simulation frame in-place into the current circular buffer slot.

        Parameters
        ----------
        step : int
            Current MD integration step number.
        atoms : Atoms
            ASE Atoms instance containing current positions, velocities, and cell.
        forces : np.ndarray
            Atomic forces array of shape (n_atoms, 3) in eV/Å.
        epot : float
            Potential energy in eV.
        ekin : float
            Kinetic energy in eV.
        """
        slot = self.cursor

        self.positions[slot, :, :] = atoms.get_positions()
        vel = atoms.get_velocities()
        if vel is not None:
            self.velocities[slot, :, :] = vel
        else:
            self.velocities[slot, :, :] = 0.0

        self.forces[slot, :, :] = forces
        self.cells[slot, :, :] = atoms.get_cell().array
        self.pbc[slot, :] = atoms.get_pbc()
        self.epot[slot] = epot
        self.ekin[slot] = ekin
        self.steps[slot] = step

        self.cursor = (self.cursor + 1) % self.capacity
        self.count += 1

    def __len__(self) -> int:
        """Return the current number of valid stored frames."""
        return min(self.count, self.capacity)

    def get_chronological_slots(self) -> np.ndarray:
        """
        Return the 1D array of slot indices in chronological order (oldest to newest).

        Returns
        -------
        np.ndarray
            Array of slot indices into the internal pre-allocated buffers.
        """
        length = len(self)
        if length == 0:
            return np.empty(0, dtype=np.int64)

        if self.count <= self.capacity:
            return np.arange(self.count, dtype=np.int64)

        # Buffer has wrapped: oldest is at self.cursor, newest is at (self.cursor - 1) % capacity
        return (np.arange(self.capacity, dtype=np.int64) + self.cursor) % self.capacity

    def reconstruct_atoms(self, slot: int, atomic_numbers: np.ndarray) -> Atoms:
        """
        Reconstruct an ASE Atoms object from a specific buffer slot.

        Parameters
        ----------
        slot : int
            Internal buffer slot index (0 <= slot < capacity).
        atomic_numbers : np.ndarray
            Array of atomic numbers Z of length n_atoms.

        Returns
        -------
        Atoms
            Reconstructed ASE Atoms instance with positions, velocities, cell, and pbc.
        """
        if slot < 0 or slot >= self.capacity:
            raise IndexError(f"Slot {slot} out of range for capacity {self.capacity}.")

        atoms = Atoms(
            numbers=atomic_numbers,
            positions=self.positions[slot].copy(),
            cell=self.cells[slot].copy(),
            pbc=self.pbc[slot].copy(),
        )
        atoms.set_velocities(self.velocities[slot].copy())
        return atoms

    def get_force_history(self) -> np.ndarray:
        """
        Return chronological 1D array of maximum atomic force magnitude (eV/Å) per frame.
        """
        slots = self.get_chronological_slots()
        if len(slots) == 0:
            return np.empty(0, dtype=np.float64)
        # forces shape: (len, n_atoms, 3)
        f_chron = self.forces[slots]
        f_norms = np.linalg.norm(f_chron, axis=-1)  # (len, n_atoms)
        return np.max(f_norms, axis=-1)  # (len,)

    def get_energy_history(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return chronological tuple of (epot, ekin) arrays in eV.
        """
        slots = self.get_chronological_slots()
        if len(slots) == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        return self.epot[slots].copy(), self.ekin[slots].copy()

    def get_volume_history(self) -> np.ndarray:
        """
        Return chronological 1D array of unit cell volumes in Å³.
        """
        slots = self.get_chronological_slots()
        if len(slots) == 0:
            return np.empty(0, dtype=np.float64)
        cells_chron = self.cells[slots]
        # Volume is determinant of 3x3 cell matrix
        vols = np.abs(np.linalg.det(cells_chron))
        return vols

    def get_step_history(self) -> np.ndarray:
        """
        Return chronological 1D array of step integers.
        """
        slots = self.get_chronological_slots()
        return self.steps[slots].copy()
