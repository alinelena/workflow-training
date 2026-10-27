"""
Tests for Stage 0 Cold Start: EOS strain series and harmonic rattles.
"""

from ase import Atoms

from workflow_training.active_learning.stages.stage0_cold_start import (
    build_stage0_subgraph,
    generate_cold_start_structures,
)


def test_generate_cold_start_structures(silicon_structure: Atoms):
    strains = [-0.10, -0.05, 0.0, 0.05, 0.10]
    n_rattles = 3
    generated = generate_cold_start_structures(
        base_atoms=silicon_structure,
        strain_range=strains,
        n_rattles=n_rattles,
        rattle_std=0.03,
    )

    # Expected count: len(strains) + n_rattles = 5 + 3 = 8
    assert len(generated) == len(strains) + n_rattles

    # Verify strain tags
    eos_configs = [a for a in generated if a.info.get("type") == "eos_strain"]
    rattle_configs = [a for a in generated if a.info.get("type") == "harmonic_rattle"]

    assert len(eos_configs) == 5
    assert len(rattle_configs) == 3

    # Check that rattler displaced atoms
    orig_pos = silicon_structure.get_positions()
    for rat in rattle_configs:
        assert not (rat.get_positions() == orig_pos).all()


def test_build_stage0_subgraph(silicon_structure: Atoms, mock_pw_code):
    sub = build_stage0_subgraph(
        structure=silicon_structure,
        pw_code=mock_pw_code,
        pseudo_family="SSSP",
    )
    assert sub is not None
