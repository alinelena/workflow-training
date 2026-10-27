"""
Example 1: Guardrailed Curriculum Active Learning WorkGraph setup with MACE and Quantum ESPRESSO.

This example illustrates how to configure and build the top-level Active Learning WorkGraph
using aiida-workgraph, STFC aiida-mlip, and Quantum ESPRESSO.
"""

from __future__ import annotations

from ase import Atoms

try:
    from aiida import load_profile, orm
    from aiida_mlip.data.model import ModelData

    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False
    orm = None
    ModelData = None

from workflow_training.active_learning.workgraph import build_guardrailed_al_workgraph


def main():
    print("=" * 70)
    print("Example 1: Guardrailed Active Learning WorkGraph Construction")
    print("=" * 70)

    # 1. Define Crystal Structure (Diamond Silicon)
    a = 5.43
    si_atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[[0.0, 0.0, 0.0], [0.25 * a, 0.25 * a, 0.25 * a]],
        cell=[
            [0.0, 0.5 * a, 0.5 * a],
            [0.5 * a, 0.0, 0.5 * a],
            [0.5 * a, 0.5 * a, 0.0],
        ],
        pbc=True,
    )
    print(f"Base structure: Silicon (2 atoms, volume = {si_atoms.get_volume():.2f} Å³)")

    # 2. Configure Curriculum and Watchdog Thresholds
    curriculum_config = {
        "temperatures": [300.0, 600.0, 900.0],  # Ascending temperature ladder (K)
        "pressure_bar": 0.0,  # Ambient isobaric pressure
        "md_steps_per_epoch": 5000,  # MD steps per exploration
        "timestep_fs": 1.0,  # MD timestep (fs)
        "max_epochs": 10,  # Prevent runaway loop
        "tri_zone": {
            "f_low": 0.8,  # Lower force bound (eV/Å): below is equilibrium
            "f_high": 8.0,  # Upper force bound (eV/Å): active learning zone
            "max_candidates_per_epoch": 20,  # Max DFT jobs per epoch
        },
        "necropsy": {
            "clash_ratio": 0.70,  # Fractional sum of covalent radii
            "force_max": 25.0,  # Force explosion ceiling (eV/Å)
            "vol_dilation": 0.20,  # Max relative volume dilation
            "cell_skew_min": 0.15,  # Min condition number of cell matrix
            "safe_ratio": 0.85,  # Cusp safety threshold
        },
        "training": {
            "energy_weight": 1.0,
            "forces_weight": 20.0,
            "stress_weight": 5.0,
            "learning_rate": 1e-4,
            "max_num_epochs": 50,
            "batch_size": 8,
            "anchor_replay_ratio": 0.20,  # Guaranteed 20% Stage 0 replay
        },
    }

    # 3. Configure Quantum ESPRESSO Parameters
    qe_parameters = {
        "ecutwfc": 60.0,
        "ecutrho": 480.0,
        "kpoints_distance": 0.03,
        "conv_thr": 1.0e-8,
        "smearing": "cold",
        "degauss": 0.02,
        "tstress": True,
        "tprnfor": True,
    }

    # 4. Build the WorkGraph
    if AIIDA_AVAILABLE:
        try:
            load_profile()
            structure_node = orm.StructureData(ase=si_atoms)
            # Codes on cluster / local
            try:
                pw_code = orm.load_code("pw@localhost")
            except Exception:
                pw_code = orm.load_code("pw-7.2@localhost")

            try:
                janus_code = orm.load_code("janus@localhost")
            except Exception:
                janus_code = orm.load_code("janus@localhost")

            pseudo_family = orm.Str("SSSP/1.3/PBE/efficiency")
            try:
                model_node = orm.load_node(label="mace_mp_medium")
            except Exception:
                model_node = orm.Str("mace_mp_medium")

            wg = build_guardrailed_al_workgraph(
                structure=structure_node,
                model=model_node,
                pw_code=pw_code,
                janus_code=janus_code,
                pseudo_family=pseudo_family,
                curriculum_config=orm.Dict(dict=curriculum_config),
                qe_parameters=orm.Dict(dict=qe_parameters),
                max_number_jobs=16,
            )
            print("Successfully constructed AiiDA WorkGraph DAG!")
            print(f"Max concurrent HPC jobs: {wg.max_number_jobs}")
            ctx_keys = list(getattr(wg, "context", getattr(wg, "ctx", {})).keys())
            print(f"Initial context keys: {ctx_keys}")
            # To submit on cluster:
            # wg.submit()
            return
        except Exception as e:
            print(f"AiiDA profile/codes not loaded ({e}); demonstrating offline build:")

    # Offline / Mock Build Demonstration
    wg_meta = build_guardrailed_al_workgraph(
        structure=si_atoms,
        model="mace_mp_foundation.model",
        pw_code="pw.x",
        janus_code="janus",
        pseudo_family="SSSP",
        curriculum_config=curriculum_config,
        qe_parameters=qe_parameters,
        max_number_jobs=16,
    )
    print("\nOffline WorkGraph Model Generated:")
    print(f"  Name: {wg_meta['name']}")
    print(f"  Max HPC Jobs: {wg_meta['max_number_jobs']}")
    print(f"  Context Keys: {wg_meta['context_keys']}")
    print(f"  Curriculum Temperatures: {wg_meta['curriculum_config']['temperatures']} K")
    print("=" * 70)


if __name__ == "__main__":
    main()
