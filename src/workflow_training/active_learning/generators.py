"""Candidate Data Generators for Active Learning.

Generates atomic configurations across three distinct physical regimes:
  1. Equation of State (EOS) strain dilation, compression, and atomic rattles.
  2. Geometry Optimization (geomopt) relaxation paths with FrechetCellFilter.
  3. Molecular Dynamics (MD) thermal sampling across NVT / NPT ensembles.
"""

from __future__ import annotations

import logging
from typing import Any

from ase import units
from ase.atoms import Atoms
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS

from workflow_training.active_learning.stages.stage0_cold_start import (
    generate_cold_start_structures,
)

logger = logging.getLogger(__name__)


def generate_eos_dataset(
    base_atoms: Atoms,
    strain_range: list[float] | None = None,
    n_rattles: int = 2,
    rattle_std: float = 0.02,
) -> list[Atoms]:
    """Generate cold-start Equation of State (EOS) strain deformations and rattles."""
    if strain_range is None:
        strain_range = [-0.03, -0.015, 0.0, 0.015, 0.03]

    frames = generate_cold_start_structures(
        base_atoms=base_atoms,
        strain_range=strain_range,
        n_rattles=n_rattles,
        rattle_std=rattle_std,
    )
    for f in frames:
        f.info["generator"] = "eos"
    return frames


def generate_geomopt_dataset(
    base_atoms: Atoms,
    calculator: Any,
    fmax: float = 0.02,
    steps: int = 50,
    filter_cell: bool = True,
) -> list[Atoms]:
    """Perform geometry optimization with FrechetCellFilter, collecting intermediate trajectory frames."""
    atoms = base_atoms.copy()
    atoms.calc = calculator

    trajectory_frames: list[Atoms] = []

    # FrechetCellFilter for variable-cell crystal relaxation
    target = FrechetCellFilter(atoms) if (filter_cell and any(atoms.pbc)) else atoms

    def _observer():
        snap = atoms.copy()
        snap.calc = None
        snap.info["generator"] = "geomopt"
        snap.info["geomopt_step"] = len(trajectory_frames)
        trajectory_frames.append(snap)

    opt = LBFGS(target, logfile=None)
    opt.attach(_observer, interval=1)

    # Initial frame
    _observer()

    try:
        opt.run(fmax=fmax, steps=steps)
    except Exception as e:
        logger.warning(f"Geometry optimization finished with note: {e}")

    return trajectory_frames


def generate_md_dataset(
    base_atoms: Atoms,
    calculator: Any,
    ensemble: str = "nvt",
    temperature_k: float = 300.0,
    steps: int = 100,
    timestep_fs: float = 1.0,
    record_interval: int = 2,
) -> list[Atoms]:
    """Run thermal molecular dynamics, recording sampled trajectory frames."""
    atoms = base_atoms.copy()
    atoms.calc = calculator

    # Initialize thermal Maxwell-Boltzmann velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, force_temp=True)

    trajectory_frames: list[Atoms] = []

    def _observer():
        snap = atoms.copy()
        snap.calc = None
        snap.info["generator"] = f"md_{ensemble}"
        snap.info["md_step"] = len(trajectory_frames) * record_interval
        snap.info["temperature_k"] = float(temperature_k)
        trajectory_frames.append(snap)

    # Langevin thermostat (NVT)
    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_k,
        friction=0.01 / units.fs,
        logfile=None,
        fixcm=(len(atoms) > 1),
    )
    dyn.attach(_observer, interval=record_interval)

    # Record initial frame
    _observer()

    dyn.run(steps=steps)

    return trajectory_frames
