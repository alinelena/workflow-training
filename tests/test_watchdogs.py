"""
Tests for NecropsyWatchdog: clash, force explosion, volume dilation, and cell skew triggers.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from workflow_training.active_learning.watchdogs import (
    NecropsyWatchdog,
    SimulationWatchdogTriggered,
)


class DummyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, fixed_forces=None):
        super().__init__()
        self.fixed_forces = fixed_forces

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        self.results["energy"] = -10.0
        if self.fixed_forces is not None:
            self.results["forces"] = self.fixed_forces
        else:
            self.results["forces"] = np.zeros((n, 3))


def test_watchdog_normal_step(silicon_structure: Atoms):
    silicon_structure.calc = DummyCalculator()
    watchdog = NecropsyWatchdog(silicon_structure)

    # Should not raise
    watchdog.step(step_idx=0)
    assert len(watchdog.buffer) == 1


def test_watchdog_clash_trigger(silicon_structure: Atoms):
    # Place two atoms extremely close (0.5 Å)
    clashed = silicon_structure.copy()
    pos = clashed.get_positions()
    pos[1] = pos[0] + [0.3, 0.0, 0.0]  # Severe clash
    clashed.set_positions(pos)
    clashed.calc = DummyCalculator()

    watchdog = NecropsyWatchdog(clashed, clash_ratio_threshold=0.70)

    with pytest.raises(SimulationWatchdogTriggered) as excinfo:
        watchdog.step(step_idx=1)

    assert excinfo.value.reason == "clash"
    assert excinfo.value.step == 1
    assert excinfo.value.metrics["min_ratio"] < 0.70


def test_watchdog_force_spike_trigger(silicon_structure: Atoms):
    n = len(silicon_structure)
    huge_forces = np.zeros((n, 3))
    huge_forces[0] = [30.0, 0.0, 0.0]  # Exceeds 25.0 eV/Å
    silicon_structure.calc = DummyCalculator(fixed_forces=huge_forces)

    watchdog = NecropsyWatchdog(silicon_structure, f_max_threshold=25.0)

    with pytest.raises(SimulationWatchdogTriggered) as excinfo:
        watchdog.step(step_idx=5)

    assert excinfo.value.reason == "force"
    assert excinfo.value.metrics["f_max"] >= 30.0


def test_watchdog_volume_dilation_trigger(silicon_structure: Atoms):
    silicon_structure.calc = DummyCalculator()
    watchdog = NecropsyWatchdog(silicon_structure, vol_dilation_threshold=0.20)

    # Dilate cell by 30%
    dilated_cell = silicon_structure.get_cell() * 1.15  # Volume expands ~1.52x
    silicon_structure.set_cell(dilated_cell, scale_atoms=False)

    with pytest.raises(SimulationWatchdogTriggered) as excinfo:
        watchdog.step(step_idx=10)

    assert excinfo.value.reason == "volume"
    assert excinfo.value.metrics["dilation"] > 0.20


def test_watchdog_cell_skew_trigger(silicon_structure: Atoms):
    silicon_structure.calc = DummyCalculator()
    watchdog = NecropsyWatchdog(silicon_structure, cell_skew_threshold=0.15)

    # Create severe volume-preserving shear (det remains unchanged, but cond(cell) < 0.15)
    orig_cell = silicon_structure.get_cell().array
    skewed_cell = orig_cell.copy()
    skewed_cell[0] += 8.0 * orig_cell[1]  # Massive volume-preserving shear

    silicon_structure.set_cell(skewed_cell, scale_atoms=False)

    with pytest.raises(SimulationWatchdogTriggered) as excinfo:
        watchdog.step(step_idx=12)

    assert excinfo.value.reason == "skew"
    assert excinfo.value.metrics["cond_ratio"] < 0.15
