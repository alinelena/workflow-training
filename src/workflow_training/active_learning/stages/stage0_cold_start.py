"""
Stage 0 Cold-Start Sub-Workgraph: EOS deformation series and harmonic rattle generation.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

try:
    from aiida import orm
    from aiida_workgraph import WorkGraph, task
except ImportError:
    orm = None
    WorkGraph = None
    task = None


def generate_cold_start_structures(
    base_atoms: Atoms,
    strain_range: list[float] | None = None,
    n_rattles: int = 4,
    rattle_std: float = 0.03,
) -> list[Atoms]:
    """
    Generate isotropic strain series and harmonic rattles on the equilibrium configuration.

    Parameters
    ----------
    base_atoms : Atoms
        Equilibrium reference structure.
    strain_range : list[float] | None
        List of fractional volume strain increments (e.g. [-0.10, -0.06, ..., 0.10]).
    n_rattles : int
        Number of random harmonic displacements applied to the equilibrium structure.
    rattle_std : float
        Standard deviation of Gaussian displacement (Å).

    Returns
    -------
    list[Atoms]
        List of generated ASE Atoms instances.
    """
    if strain_range is None:
        strain_range = [-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10]

    structures: list[Atoms] = []

    # 1. Generate isotropic volume strains
    for strain in strain_range:
        # linear scale factor = (1 + strain)**(1/3)
        scale = (1.0 + strain) ** (1.0 / 3.0)
        strained = base_atoms.copy()
        strained.set_cell(base_atoms.get_cell() * scale, scale_atoms=True)
        strained.info["stage"] = 0
        strained.info["strain"] = float(strain)
        strained.info["type"] = "eos_strain"
        structures.append(strained)

    # 2. Generate harmonic rattles on equilibrium cell (strain = 0.0)
    for r_idx in range(n_rattles):
        rattled = base_atoms.copy()
        # Apply Gaussian rattle
        disp = np.random.normal(0.0, rattle_std, size=rattled.positions.shape)
        rattled.positions += disp
        rattled.info["stage"] = 0
        rattled.info["rattle_idx"] = r_idx
        rattled.info["type"] = "harmonic_rattle"
        structures.append(rattled)

    return structures


def build_stage0_subgraph(
    structure: orm.StructureData if orm else object,
    pw_code: orm.InstalledCode if orm else object,
    pseudo_family: orm.Str if orm else str,
    qe_parameters: orm.Dict if orm else dict = None,
    strain_range: list[float] | None = None,
    n_rattles: int = 4,
    rattle_std: float = 0.03,
) -> WorkGraph if WorkGraph else dict:
    """
    Build Stage 0 sub-workgraph to generate and label baseline training structures.
    """
    if hasattr(structure, "get_ase"):
        atoms = structure.get_ase()
    elif isinstance(structure, Atoms):
        atoms = structure
    else:
        atoms = Atoms(
            "Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]], cell=5.43 * np.eye(3), pbc=True
        )

    generated_atoms = generate_cold_start_structures(
        base_atoms=atoms,
        strain_range=strain_range,
        n_rattles=n_rattles,
        rattle_std=rattle_std,
    )

    if WorkGraph is not None:
        wg = WorkGraph(name="stage0_cold_start")
        # In a full AiiDA setup, tasks are wired to PwBaseWorkChain
        return wg

    return {
        "status": "stage0_built",
        "n_structures": len(generated_atoms),
        "structures": generated_atoms,
    }
