"""
Pytest fixtures and mock objects for offline testing.
"""

import pytest
from ase import Atoms


@pytest.fixture(scope="session", autouse=True)
def load_aiida_profile():
    try:
        from aiida import load_profile
        from aiida.manage.configuration import get_profile

        if get_profile() is None:
            load_profile("janus_local")
    except Exception:
        pass


@pytest.fixture
def silicon_structure() -> Atoms:
    """Return standard Diamond Silicon 2-atom unit cell."""
    a = 5.43
    return Atoms(
        symbols=["Si", "Si"],
        positions=[[0.0, 0.0, 0.0], [0.25 * a, 0.25 * a, 0.25 * a]],
        cell=[
            [0.0, 0.5 * a, 0.5 * a],
            [0.5 * a, 0.0, 0.5 * a],
            [0.5 * a, 0.5 * a, 0.0],
        ],
        pbc=True,
    )


@pytest.fixture
def curriculum_config() -> dict:
    """Standard curriculum configuration."""
    return {
        "temperatures": [300.0, 600.0, 900.0],
        "pressure_bar": 0.0,
        "md_steps_per_epoch": 50,
        "timestep_fs": 1.0,
        "max_epochs": 5,
        "tri_zone": {
            "f_low": 0.8,
            "f_high": 8.0,
            "max_candidates_per_epoch": 10,
        },
        "necropsy": {
            "clash_ratio": 0.70,
            "force_max": 25.0,
            "vol_dilation": 0.20,
            "cell_skew_min": 0.15,
            "safe_ratio": 0.85,
        },
        "training": {
            "energy_weight": 1.0,
            "forces_weight": 20.0,
            "stress_weight": 5.0,
            "learning_rate": 1e-4,
            "max_num_epochs": 10,
            "batch_size": 4,
            "anchor_replay_ratio": 0.20,
        },
    }


@pytest.fixture
def mock_pw_code():
    """Mock Quantum ESPRESSO code node."""

    class MockCode:
        label = "pw-7.2@localhost"
        pk = 101

    return MockCode()


@pytest.fixture
def mock_janus_code():
    """Mock Janus-Core code node."""

    class MockCode:
        label = "janus@localhost"
        pk = 102

    return MockCode()


@pytest.fixture
def mock_model():
    """Mock MACE ModelData node."""

    class MockModel:
        model_type = "mace_mp"
        pk = 201

    return MockModel()
