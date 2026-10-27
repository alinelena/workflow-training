"""
Example 2: In-flight Multi-Modal Watchdogs and Post-Mortem Necropsy-Rewind.

Demonstrates:
1. Low-latency TrajectoryRingBuffer (< 15 µs).
2. Real-time physical watchdogs (clash, force spike, volume dilation, cell skew).
3. Post-mortem reverse Z-score inflection cusp extraction (strictly discarding corpse frame).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from workflow_training.active_learning.necropsy import find_inflection_cusp
from workflow_training.active_learning.ring_buffer import TrajectoryRingBuffer
from workflow_training.active_learning.watchdogs import (
    NecropsyWatchdog,
    SimulationWatchdogTriggered,
)


class DivergingToyCalculator(Calculator):
    """
    Toy calculator simulating an unstable MLIP trajectory that experiences
    an extrapolation hole after step 15, culminating in a force explosion and clash.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.step_count = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        self.step_count += 1

        if self.step_count < 15:
            # Stable equilibrium thermal vibrations (~0.4 eV/Å)
            forces = np.random.normal(0.0, 0.4, size=(n, 3))
            energy = -15.0 + 0.05 * np.random.randn()
        elif self.step_count < 18:
            # Pre-collapse non-linear divergence (inflection cusp region ~2.5 eV/Å)
            forces = np.full((n, 3), 1.5)
            energy = -12.0
        else:
            # Catastrophic extrapolation explosion (> 30 eV/Å and atomic clash)
            forces = np.full((n, 3), 25.0)
            energy = 150.0

        self.results["energy"] = energy
        self.results["forces"] = forces


def main():
    print("=" * 70)
    print("Example 2: Watchdogs & Necropsy-Rewind Demonstration")
    print("=" * 70)

    # 1. Initialize Silicon System
    a = 5.43
    atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[[0.0, 0.0, 0.0], [0.25 * a, 0.25 * a, 0.25 * a]],
        cell=[
            [0.0, 0.5 * a, 0.5 * a],
            [0.5 * a, 0.0, 0.5 * a],
            [0.5 * a, 0.5 * a, 0.0],
        ],
        pbc=True,
    )
    atoms.calc = DivergingToyCalculator()

    # 2. Attach NecropsyWatchdog with Ring Buffer
    buffer_cap = 50
    buffer = TrajectoryRingBuffer(capacity=buffer_cap, n_atoms=len(atoms))
    watchdog = NecropsyWatchdog(
        atoms=atoms,
        buffer=buffer,
        buffer_capacity=buffer_cap,
        f_max_threshold=25.0,  # Halt if force > 25 eV/Å
        clash_ratio_threshold=0.70,  # Halt if r_ij < 0.70 (R_i + R_j)
        vol_dilation_threshold=0.20,  # Halt if |V-V0|/V0 > 20%
        cell_skew_threshold=0.15,  # Halt if cond(cell) < 0.15
    )

    print(f"Watchdog attached: Buffer capacity = {buffer_cap} frames")
    print("Running time integration loop...")

    # 3. Step through trajectory until watchdog trips
    crashed = False
    for step in range(30):
        # In a real run, dyn.step() would displace atoms
        if step >= 18:
            # Simulate atomic clash at step 18
            pos = atoms.get_positions()
            pos[1] = pos[0] + [0.4, 0.0, 0.0]  # Severe clash (< 0.70 R_cov)
            atoms.set_positions(pos)

        try:
            watchdog.step(step_idx=step)
            f_norm = float(np.max(np.linalg.norm(atoms.get_forces(), axis=-1)))
            print(f"  Step {step:02d}: Normal integration (max ||F|| = {f_norm:.2f} eV/Å)")
        except SimulationWatchdogTriggered as exc:
            print(f"\n[!] WATCHDOG TRIPPED at step {exc.step}!")
            print(f"    Reason: {exc.reason}")
            print(f"    Metrics: {exc.metrics}")
            crashed = True
            break

    assert crashed, "Expected watchdog to trip on unphysical divergence."

    # 4. Execute Necropsy-Rewind
    print("\nExecuting Post-Mortem Necropsy Engine:")
    print("  -> Terminal crash frame is unphysical; strictly discarding from DFT queue.")
    necropsy_res = find_inflection_cusp(
        buffer=watchdog.buffer,
        atomic_numbers=atoms.get_atomic_numbers(),
        safe_ratio=0.85,
        baseline_sigma_mult=3.5,
    )

    print(f"  Discarded corpse frame: step {necropsy_res['discarded_crash_step']}")
    print(f"  Extracted physical cusp: step {necropsy_res['cusp_step']}")
    print(f"  Cusp max force: {necropsy_res['cusp_max_force']:.2f} eV/Å")
    print(f"  Safe for DFT submission: {necropsy_res['is_safe_for_dft']}")
    print(f"  Diagnosis reason: {necropsy_res['reason']}")

    # 5. Verify distance of extracted cusp
    cusp_atoms = necropsy_res["cusp_atoms"]
    dists = cusp_atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    min_dist = float(np.min(dists))
    print(f"  Minimum interatomic separation in cusp: {min_dist:.2f} Å (Safe physical geometry)")
    print("=" * 70)


if __name__ == "__main__":
    main()
