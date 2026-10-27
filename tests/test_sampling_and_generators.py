"""Unit tests for active learning candidate generators, descriptor-space FPS sampling,

and before/after performance comparison.
"""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT

from workflow_training.active_learning.generators import (
    generate_eos_dataset,
    generate_geomopt_dataset,
    generate_md_dataset,
)
from workflow_training.active_learning.plot_comparison import (
    compute_comparative_statistics,
    evaluate_frames_with_models,
)
from workflow_training.active_learning.sampling import (
    extract_frame_descriptors,
    farthest_point_sampling,
    sample_trajectory_frames,
)


def test_farthest_point_sampling_logic():
    """Verify FPS selects maximally distant points in feature space."""
    # 4 collinear points along x axis: 0, 1, 2, 10
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    # Seed 0 selects index 0
    # Next farthest should be index 3 (dist 10)
    # Next farthest should be index 1 or 2 (dist ~ 8-9)
    selected = farthest_point_sampling(points, n_samples=3, initial_idx=0)
    assert len(selected) == 3
    assert selected[0] == 0
    assert selected[1] == 3
    assert selected[2] in (1, 2)


def test_sample_trajectory_frames_methods():
    """Test various sampling strategies on synthetic trajectory frames."""
    si = bulk("Si", "diamond", a=5.43)
    frames = []
    for strain in np.linspace(-0.05, 0.05, 12):
        s = si.copy()
        s.set_cell(s.get_cell() * (1.0 + strain), scale_atoms=True)
        frames.append(s)

    # 1. Stride sampling
    stride_sampled = sample_trajectory_frames(frames, n_samples=4, method="stride")
    assert len(stride_sampled) == 4

    # 2. Random sampling
    random_sampled = sample_trajectory_frames(frames, n_samples=3, method="random")
    assert len(random_sampled) == 3

    # 3. All sampling
    all_sampled = sample_trajectory_frames(frames, n_samples=2, method="all")
    assert len(all_sampled) == 12

    # 4. FPS sampling (using fallback geometric descriptor)
    fps_sampled = sample_trajectory_frames(frames, n_samples=4, method="fps", calculator=None)
    assert len(fps_sampled) == 4
    for f in fps_sampled:
        assert f.info["sample_method"] == "fps"
        assert "original_trajectory_index" in f.info


def test_extract_frame_descriptors_fallback():
    """Verify descriptor extraction works smoothly even without MLIP."""
    si = bulk("Si", "diamond", a=5.43)
    frames = [si.copy(), si.copy()]
    descs = extract_frame_descriptors(frames, calculator=None)
    assert descs.shape[0] == 2
    assert descs.shape[1] > 0


def test_generate_eos_dataset():
    """Verify EOS dataset generation produces strained configurations."""
    si = bulk("Si", "diamond", a=5.43)
    frames = generate_eos_dataset(si, strain_range=[-0.02, 0.0, 0.02], n_rattles=1)
    # 3 strains + 1 rattle = 4 frames
    assert len(frames) == 4
    assert all(f.info.get("generator") == "eos" for f in frames)


def test_generate_geomopt_dataset_with_emt():
    """Verify geometry optimization generator records relaxation steps."""
    al = bulk("Al", "fcc", a=4.05)
    # Rattle to create nonzero forces
    al.rattle(stdev=0.05, seed=42)
    frames = generate_geomopt_dataset(al, calculator=EMT(), fmax=0.05, steps=5)
    assert len(frames) >= 2
    assert frames[0].info["generator"] == "geomopt"
    assert frames[-1].info["generator"] == "geomopt"


def test_generate_md_dataset_with_emt():
    """Verify thermal MD generator records trajectory snapshots."""
    al = bulk("Al", "fcc", a=4.05) * (2, 1, 1)
    frames = generate_md_dataset(
        al,
        calculator=EMT(),
        ensemble="nvt",
        temperature_k=150.0,
        steps=6,
        record_interval=2,
    )
    # 0, 2, 4, 6 -> 4 frames
    assert len(frames) >= 3
    assert all("md" in f.info["generator"] for f in frames)


def test_comparative_statistics_calculation():
    """Verify comparative statistics compute RMSE, MAE, R², and reductions accurately."""
    al = bulk("Al", "fcc", a=4.05)
    frames = []
    calc = EMT()

    for d in [0.0, 0.02, -0.02]:
        f = al.copy()
        f.set_cell(f.get_cell() * (1.0 + d), scale_atoms=True)
        f.calc = calc
        f.info["REF_energy"] = float(f.get_potential_energy())
        f.arrays["REF_forces"] = np.array(f.get_forces())
        f.info["REF_stress"] = np.array(f.get_stress(voigt=False)).flatten()
        f.calc = None
        frames.append(f)

    # Evaluate with two dummy calculators (EMT and EMT with slight perturbation)
    class PerturbedEMT(EMT):
        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=()):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] += 0.01

    eval_frames = evaluate_frames_with_models(frames, calc_before=PerturbedEMT(), calc_after=EMT())
    assert len(eval_frames) == 3

    stats = compute_comparative_statistics(eval_frames)
    assert "before" in stats
    assert "after" in stats
    assert "improvements" in stats
    assert stats["after"]["absolute_energy_mev_atom"]["rmse"] < 1e-4
