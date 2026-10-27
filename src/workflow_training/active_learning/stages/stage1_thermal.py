"""
Stage 1 Thermal Exploration: Guarded NPT MD runner with Necropsy-Rewind and tri-zone candidate selection.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from workflow_training.active_learning.necropsy import find_inflection_cusp
from workflow_training.active_learning.ring_buffer import TrajectoryRingBuffer
from workflow_training.active_learning.watchdogs import (
    NecropsyWatchdog,
    SimulationWatchdogTriggered,
)

try:
    from aiida import orm
    from aiida_workgraph import task
except ImportError:
    orm = None
    task = None


def _calc_or_func(func):
    if task is not None and hasattr(task, "calcfunction"):
        try:
            func.task = task.calcfunction(func)
        except Exception:
            func.task = None
    return func


@_calc_or_func
def run_guarded_md_task(
    structure: orm.StructureData if orm else object,
    model: object,
    temperature: orm.Float if orm else float,
    pressure_bar: orm.Float if orm else float,
    md_steps: orm.Int if orm else int,
    timestep_fs: orm.Float if orm else float,
    tri_zone_config: orm.Dict if orm else dict,
    necropsy_config: orm.Dict if orm else dict,
    calculator=None,
) -> dict:
    """
    Execute guarded MD simulation with NecropsyWatchdog.

    Shields AiiDA tasks from catastrophic failures by handling SimulationWatchdogTriggered
    internally and extracting the physical pre-collapse cusp via reverse Z-score analysis.
    """
    if hasattr(structure, "get_ase"):
        atoms = structure.get_ase()
    elif isinstance(structure, Atoms):
        atoms = structure.copy()
    else:
        atoms = Atoms(
            "Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]], cell=5.43 * np.eye(3), pbc=True
        )

    temp_k = float(temperature.value) if hasattr(temperature, "value") else float(temperature)
    n_steps = int(md_steps.value) if hasattr(md_steps, "value") else int(md_steps)
    dt = float(timestep_fs.value) if hasattr(timestep_fs, "value") else float(timestep_fs)

    tz_cfg = (
        tri_zone_config.get_dict()
        if hasattr(tri_zone_config, "get_dict")
        else dict(tri_zone_config)
    )
    f_low = float(tz_cfg.get("f_low", 0.8))
    f_high = float(tz_cfg.get("f_high", 8.0))
    max_cand = int(tz_cfg.get("max_candidates_per_epoch", 20))

    nec_cfg = (
        necropsy_config.get_dict()
        if hasattr(necropsy_config, "get_dict")
        else dict(necropsy_config)
    )
    clash_ratio = float(nec_cfg.get("clash_ratio", 0.70))
    force_max = float(nec_cfg.get("force_max", 25.0))
    vol_dilation = float(nec_cfg.get("vol_dilation", 0.20))
    cell_skew = float(nec_cfg.get("cell_skew_min", 0.15))
    safe_ratio = float(nec_cfg.get("safe_ratio", 0.85))

    # Initialize buffer and watchdog
    buffer_cap = min(max(50, n_steps // 2), 200)
    buffer = TrajectoryRingBuffer(capacity=buffer_cap, n_atoms=len(atoms))
    watchdog = NecropsyWatchdog(
        atoms=atoms,
        buffer=buffer,
        buffer_capacity=buffer_cap,
        f_max_threshold=force_max,
        clash_ratio_threshold=clash_ratio,
        vol_dilation_threshold=vol_dilation,
        cell_skew_threshold=cell_skew,
    )

    if calculator is not None:
        atoms.calc = calculator

    # Attach velocities
    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_k)
    except Exception:
        pass

    # Setup dynamics if calculator is attached
    status = "completed"
    crash_reason = None
    crash_metrics = {}
    cusp_result = None

    if atoms.calc is not None:
        dyn = Langevin(atoms, timestep=dt * units.fs, temperature_K=temp_k, friction=0.01)
        watchdog.dyn = dyn

        try:
            for step_i in range(n_steps):
                dyn.step()
                watchdog.step(step_idx=step_i)
        except SimulationWatchdogTriggered as exc:
            # Internal Exception Shield: catch watchdog exception and extract cusp
            status = "crashed"
            crash_reason = exc.reason
            crash_metrics = exc.metrics
            cusp_result = find_inflection_cusp(
                buffer=watchdog.buffer,
                atomic_numbers=atoms.get_atomic_numbers(),
                safe_ratio=safe_ratio,
            )
    else:
        # Dry-run / mock simulation when no calculator attached
        # Record initial configuration into buffer
        watchdog.step(step_idx=0)

    # Candidate frame selection (tri-zone force filtering)
    candidates: list[Atoms] = []
    chron_slots = buffer.get_chronological_slots()
    f_hist = buffer.get_force_history()
    atomic_numbers = atoms.get_atomic_numbers()

    # Avoid inspecting the terminal crashed slot if crashed
    valid_range = len(chron_slots) - 1 if status == "crashed" else len(chron_slots)

    candidate_slots = []
    for idx in range(valid_range):
        f_val = f_hist[idx]
        if f_low <= f_val <= f_high:
            candidate_slots.append(chron_slots[idx])

    # Temporal uniform subsampling if candidates exceed quota
    if len(candidate_slots) > max_cand:
        sub_indices = np.linspace(0, len(candidate_slots) - 1, max_cand, dtype=int)
        candidate_slots = [candidate_slots[i] for i in sub_indices]

    for slot in candidate_slots:
        cand_atoms = buffer.reconstruct_atoms(slot, atomic_numbers)
        candidates.append(cand_atoms)

    # Convert candidates and cusp to StructureData if AiiDA is active
    if orm is not None:
        candidate_nodes = [orm.StructureData(ase=a) for a in candidates]
        cusp_node = (
            orm.StructureData(ase=cusp_result["cusp_atoms"])
            if (cusp_result and cusp_result["is_safe_for_dft"])
            else None
        )
        return {
            "status": orm.Str(status),
            "candidates": candidate_nodes,
            "cusp_structure": cusp_node,
            "metrics": orm.Dict(
                dict={
                    "status": status,
                    "n_candidates": len(candidate_nodes),
                    "crash_reason": crash_reason,
                    "crash_metrics": crash_metrics,
                }
            ),
        }

    return {
        "status": status,
        "candidates": candidates,
        "cusp_structure": cusp_result["cusp_atoms"]
        if (cusp_result and cusp_result["is_safe_for_dft"])
        else None,
        "metrics": {
            "status": status,
            "n_candidates": len(candidates),
            "crash_reason": crash_reason,
            "crash_metrics": crash_metrics,
        },
    }
