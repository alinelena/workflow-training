"""
Real-time physical safety watchdogs for molecular dynamics trajectories.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from workflow_training.active_learning.ring_buffer import TrajectoryRingBuffer


class SimulationWatchdogTriggered(RuntimeError):
    """
    Raised when an in-flight molecular dynamics watchdog trips on an unphysical instability.

    Attributes
    ----------
    reason : str
        Failure cause: 'clash', 'force', 'volume', or 'skew'.
    step : int
        MD integration step where the failure occurred.
    metrics : dict
        Quantitative measurements at the trip point.
    """

    def __init__(self, reason: str, step: int, metrics: dict):
        self.reason: str = reason
        self.step: int = step
        self.metrics: dict = metrics
        super().__init__(f"Simulation watchdog tripped on '{reason}' at step {step}: {metrics}")


class NecropsyWatchdog:
    """
    ASE dynamics observer hook monitoring physical sanity at every MD step.

    Attaches to an ASE Dynamics object via dyn.attach(watchdog.step) and stores
    trajectory history into a circular TrajectoryRingBuffer.
    """

    def __init__(
        self,
        atoms: Atoms,
        dyn=None,
        buffer: TrajectoryRingBuffer | None = None,
        buffer_capacity: int = 100,
        f_max_threshold: float = 25.0,
        clash_ratio_threshold: float = 0.70,
        vol_dilation_threshold: float = 0.20,
        cell_skew_threshold: float = 0.15,
    ):
        self.atoms: Atoms = atoms
        self.dyn = dyn
        self.n_atoms: int = len(atoms)
        self.atomic_numbers: np.ndarray = atoms.get_atomic_numbers()
        self.covalent_radii: np.ndarray = covalent_radii[self.atomic_numbers]

        self.buffer_capacity: int = buffer_capacity
        self.buffer: TrajectoryRingBuffer = (
            buffer if buffer is not None else TrajectoryRingBuffer(buffer_capacity, self.n_atoms)
        )

        self.f_max_threshold: float = f_max_threshold
        self.clash_ratio_threshold: float = clash_ratio_threshold
        self.vol_dilation_threshold: float = vol_dilation_threshold
        self.cell_skew_threshold: float = cell_skew_threshold

        # Reference volume
        cell = atoms.get_cell().array
        self.v0: float = abs(float(np.linalg.det(cell))) if any(atoms.get_pbc()) else 1.0

    def step(self, step_idx: int | None = None) -> None:
        """
        Evaluate physical criteria and record frame into circular buffer.
        Called at every MD step by dyn.attach() or manual loop.
        """
        if step_idx is None:
            step_idx = self.dyn.nsteps if self.dyn is not None else self.buffer.count

        # Extract quantities
        if self.atoms.calc is not None:
            forces = self.atoms.get_forces()
            try:
                epot = float(self.atoms.get_potential_energy())
            except Exception:
                epot = 0.0
            try:
                ekin = float(self.atoms.get_kinetic_energy())
            except Exception:
                ekin = 0.0
        else:
            forces = self.atoms.arrays.get("forces", np.zeros((self.n_atoms, 3), dtype=np.float64))
            epot = self.atoms.info.get("energy", 0.0)
            try:
                ekin = float(self.atoms.get_kinetic_energy())
            except Exception:
                ekin = 0.0

        # Append to circular ring buffer
        self.buffer.append(step_idx, self.atoms, forces, epot, ekin)

        # 1. Evaluate atomic clashes
        is_clash, min_ratio, pair = self._eval_clash()
        if is_clash:
            self._trigger_halt(
                "clash",
                step_idx,
                {
                    "min_ratio": min_ratio,
                    "threshold": self.clash_ratio_threshold,
                    "pair": pair,
                    "step": step_idx,
                },
            )

        # 2. Evaluate force explosion
        f_norms = np.linalg.norm(forces, axis=-1)
        f_max = float(np.max(f_norms)) if len(f_norms) > 0 else 0.0
        if f_max > self.f_max_threshold:
            self._trigger_halt(
                "force",
                step_idx,
                {
                    "f_max": f_max,
                    "threshold": self.f_max_threshold,
                    "step": step_idx,
                },
            )

        # 3. Evaluate unit cell volume dilation (if periodic)
        if any(self.atoms.get_pbc()):
            curr_vol = abs(float(np.linalg.det(self.atoms.get_cell().array)))
            if self.v0 > 1e-6:
                dilation = abs(curr_vol - self.v0) / self.v0
                if dilation > self.vol_dilation_threshold:
                    self._trigger_halt(
                        "volume",
                        step_idx,
                        {
                            "volume": curr_vol,
                            "v0": self.v0,
                            "dilation": dilation,
                            "threshold": self.vol_dilation_threshold,
                            "step": step_idx,
                        },
                    )

            # 4. Evaluate cell shear / skew collapse (condition number)
            cell_matrix = self.atoms.get_cell().array
            singular_vals = np.linalg.svd(cell_matrix, compute_uv=False)
            if singular_vals[-1] > 1e-9:
                cond_ratio = float(singular_vals[-1] / singular_vals[0])
            else:
                cond_ratio = 0.0

            if cond_ratio < self.cell_skew_threshold:
                self._trigger_halt(
                    "skew",
                    step_idx,
                    {
                        "cond_ratio": cond_ratio,
                        "threshold": self.cell_skew_threshold,
                        "step": step_idx,
                    },
                )

    def _eval_clash(self) -> tuple[bool, float, tuple[int, int]]:
        """
        Check for interatomic clashes using minimum image convention.

        Returns
        -------
        tuple[bool, float, tuple[int, int]]
            (is_clashed, minimum_covalent_ratio, (atom_i, atom_j))
        """
        if self.n_atoms <= 1:
            return False, 1.0, (0, 0)

        dist_matrix = self.atoms.get_all_distances(mic=any(self.atoms.get_pbc()))
        # Cov sum matrix: R_cov_i + R_cov_j
        cov_sum = self.covalent_radii[:, None] + self.covalent_radii[None, :]

        # Ignore self-interactions (diagonal)
        np.fill_diagonal(dist_matrix, np.inf)

        ratio_matrix = dist_matrix / (cov_sum + 1e-12)
        min_idx = np.unravel_index(np.argmin(ratio_matrix), ratio_matrix.shape)
        min_ratio = float(ratio_matrix[min_idx])

        is_clashed = min_ratio < self.clash_ratio_threshold
        return is_clashed, min_ratio, (int(min_idx[0]), int(min_idx[1]))

    def _trigger_halt(self, reason: str, step: int, metrics: dict) -> None:
        """Raise watchdog exception to abort simulation loop."""
        raise SimulationWatchdogTriggered(reason, step, metrics)
