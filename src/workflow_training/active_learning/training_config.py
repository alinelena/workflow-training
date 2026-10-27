"""
MACE fine-tuning configuration preparation for STFC aiida-mlip and janus-core.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

try:
    from aiida import orm
    from aiida_workgraph import task
except ImportError:
    # Offline / Mock fallback
    orm = None
    task = None

try:
    from aiida_mlip.data.config import JanusConfigfile
except ImportError:
    # Fallback to SinglefileData if aiida-mlip data type not registered
    if orm is not None:
        JanusConfigfile = orm.SinglefileData
    else:
        JanusConfigfile = None


def _calc_or_func(func):
    """Decorator applying @task.calcfunction when AiiDA is loaded, else plain func."""
    if task is not None and hasattr(task, "calcfunction"):
        try:
            func.task = task.calcfunction(func)
        except Exception:
            func.task = None
    return func


@_calc_or_func
def prepare_mace_training_config(
    dataset_file: orm.SinglefileData if orm else object,
    base_model: object,
    training_params: orm.Dict if orm else dict,
) -> JanusConfigfile:
    """
    Generate JanusConfigfile containing loss weights and hyperparameters for mlip.train.

    Enforces headless execution by setting tracker: false to prevent background CodeCarbon
    and pynvml warnings.
    """
    params = (
        training_params.get_dict()
        if hasattr(training_params, "get_dict")
        else dict(training_params)
    )

    config_dict = {
        "model_type": "mace_mp",
        "energy_weight": float(params.get("energy_weight", 1.0)),
        "forces_weight": float(params.get("forces_weight", 20.0)),
        "stress_weight": float(params.get("stress_weight", 5.0)),
        "learning_rate": float(params.get("learning_rate", 1e-4)),
        "max_num_epochs": int(params.get("max_num_epochs", 50)),
        "batch_size": int(params.get("batch_size", 8)),
        "valid_fraction": float(params.get("valid_fraction", 0.1)),
        "loss": "huber",
        "huber_delta": 0.01,
        "optimizer": "adamw",
        "scheduler": "ReduceLROnPlateau",
        "lr_scheduler_gamma": 0.6,
        "lr_scheduler_patience": 6,
        "tracker": False,  # STFC Janus-Core Headless Execution Rule
    }

    # Write temporary YAML config
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False)
        temp_path = f.name

    can_use_orm = False
    if orm is not None:
        try:
            from aiida.manage.configuration import get_profile

            can_use_orm = get_profile() is not None
        except Exception:
            can_use_orm = False

    try:
        if can_use_orm and JanusConfigfile is not None:
            config_node = JanusConfigfile(file=temp_path)
        elif can_use_orm:
            config_node = orm.SinglefileData(file=temp_path)
        else:
            config_node = config_dict
    finally:
        if can_use_orm:
            Path(temp_path).unlink(missing_ok=True)

    return config_node
