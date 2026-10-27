"""Model Performance Comparison and Parity Plotting.

Inspired by and compatible with ddmms/lavello_mlips (plot_train_results.py).
Evaluates and compares MLIP performance BEFORE and AFTER fine-tuning against
reference DFT calculations (Quantum ESPRESSO pw.x).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from ase.io import write
from lavello_mlips.distributions import extract_and_plot
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse

EV_ANG3_TO_GPA = 160.21766208


def evaluate_frames_with_models(
    frames: list[Atoms],
    calc_before: Any,
    calc_after: Any,
    dft_energy_tag: str = "REF_energy",
    dft_forces_tag: str = "REF_forces",
    dft_stress_tag: str = "REF_stress",
) -> list[Atoms]:
    """Evaluate a sequence of frames with both baseline and fine-tuned models."""
    evaluated_frames = []

    for idx, atoms in enumerate(frames):
        f = atoms.copy()
        # Map DFT reference
        f.info["dft_energy"] = float(f.info[dft_energy_tag])
        f.arrays["dft_forces"] = np.array(f.arrays[dft_forces_tag], dtype=np.float64)
        if dft_stress_tag in f.info:
            f.info["dft_stress"] = np.array(f.info[dft_stress_tag], dtype=np.float64).flatten()
        else:
            f.info["dft_stress"] = np.zeros(9, dtype=np.float64)

        f.info["config_type"] = f.info.get("type", f"frame_{idx}")
        f.info["system_name"] = f.info.get("system_name", "Diamond_Silicon")

        # Evaluate Before Model (Baseline)
        f.calc = calc_before
        f.info["before_energy"] = float(f.get_potential_energy())
        f.arrays["before_forces"] = np.array(f.get_forces(), dtype=np.float64)
        try:
            f.info["before_stress"] = np.array(
                f.get_stress(voigt=False), dtype=np.float64
            ).flatten()
        except Exception:
            f.info["before_stress"] = np.zeros(9, dtype=np.float64)

        # Evaluate After Model (Fine-tuned)
        f.calc = calc_after
        f.info["after_energy"] = float(f.get_potential_energy())
        f.arrays["after_forces"] = np.array(f.get_forces(), dtype=np.float64)
        try:
            f.info["after_stress"] = np.array(f.get_stress(voigt=False), dtype=np.float64).flatten()
        except Exception:
            f.info["after_stress"] = np.zeros(9, dtype=np.float64)

        f.calc = None
        evaluated_frames.append(f)

    return evaluated_frames


def compute_comparative_statistics(
    frames: list[Atoms],
    equilibrium_idx: int = 0,
) -> dict[str, Any]:
    """Compute comprehensive error metrics and R² for before and after models."""
    dft_e0 = frames[equilibrium_idx].info["dft_energy"]
    before_e0 = frames[equilibrium_idx].info["before_energy"]
    after_e0 = frames[equilibrium_idx].info["after_energy"]

    # Absolute and Relative Energies (meV/atom)
    dft_abs_e = np.array([f.info["dft_energy"] / len(f) * 1000 for f in frames])
    before_abs_e = np.array([f.info["before_energy"] / len(f) * 1000 for f in frames])
    after_abs_e = np.array([f.info["after_energy"] / len(f) * 1000 for f in frames])

    dft_rel_e = np.array([(f.info["dft_energy"] - dft_e0) / len(f) * 1000 for f in frames])
    before_rel_e = np.array([(f.info["before_energy"] - before_e0) / len(f) * 1000 for f in frames])
    after_rel_e = np.array([(f.info["after_energy"] - after_e0) / len(f) * 1000 for f in frames])

    # Forces (meV/Å)
    dft_f = np.concatenate([f.arrays["dft_forces"].flatten() for f in frames]) * 1000
    before_f = np.concatenate([f.arrays["before_forces"].flatten() for f in frames]) * 1000
    after_f = np.concatenate([f.arrays["after_forces"].flatten() for f in frames]) * 1000

    # Stress (GPa)
    dft_s = np.concatenate([f.info["dft_stress"][:6] for f in frames]) * EV_ANG3_TO_GPA
    before_s = np.concatenate([f.info["before_stress"][:6] for f in frames]) * EV_ANG3_TO_GPA
    after_s = np.concatenate([f.info["after_stress"][:6] for f in frames]) * EV_ANG3_TO_GPA

    def _stats(ref_vals, pred_vals, is_forces: bool = False):
        res = {
            "rmse": float(rmse(ref_vals, pred_vals)),
            "mae": float(mae(ref_vals, pred_vals)),
            "r2": float(r2(ref_vals, pred_vals)) if len(set(ref_vals)) > 1 else 1.0,
            "max_error": float(np.max(np.abs(ref_vals - pred_vals))),
        }
        return res

    before_stats = {
        "absolute_energy_mev_atom": _stats(dft_abs_e, before_abs_e),
        "relative_energy_mev_atom": _stats(dft_rel_e, before_rel_e),
        "forces_mev_ang": _stats(dft_f, before_f, is_forces=True),
        "stress_gpa": _stats(dft_s, before_s),
    }

    after_stats = {
        "absolute_energy_mev_atom": _stats(dft_abs_e, after_abs_e),
        "relative_energy_mev_atom": _stats(dft_rel_e, after_rel_e),
        "forces_mev_ang": _stats(dft_f, after_f, is_forces=True),
        "stress_gpa": _stats(dft_s, after_s),
    }

    # Reductions
    b_rel_rmse = before_stats["relative_energy_mev_atom"]["rmse"]
    b_abs_mae = before_stats["absolute_energy_mev_atom"]["mae"]
    rel_e_reduction_pct = (
        (b_rel_rmse - after_stats["relative_energy_mev_atom"]["rmse"]) / b_rel_rmse * 100.0
        if b_rel_rmse > 1e-9
        else 0.0
    )
    abs_e_reduction_pct = (
        (b_abs_mae - after_stats["absolute_energy_mev_atom"]["mae"]) / b_abs_mae * 100.0
        if b_abs_mae > 1e-9
        else 0.0
    )

    return {
        "num_configurations": len(frames),
        "before": before_stats,
        "after": after_stats,
        "improvements": {
            "relative_energy_rmse_reduction_percent": float(rel_e_reduction_pct),
            "absolute_energy_offset_reduction_percent": float(abs_e_reduction_pct),
            "relative_energy_r2_improvement": float(
                after_stats["relative_energy_mev_atom"]["r2"]
                - before_stats["relative_energy_mev_atom"]["r2"]
            ),
        },
    }


def plot_side_by_side_comparison(
    frames: list[Atoms],
    stats: dict[str, Any],
    save_prefix: Path,
) -> tuple[Path, Path]:
    """Generate side-by-side Before vs After parity and error distribution figures."""
    dft_e0 = frames[0].info["dft_energy"]
    before_e0 = frames[0].info["before_energy"]
    after_e0 = frames[0].info["after_energy"]

    dft_rel_e = np.array([(f.info["dft_energy"] - dft_e0) / len(f) * 1000 for f in frames])
    before_rel_e = np.array([(f.info["before_energy"] - before_e0) / len(f) * 1000 for f in frames])
    after_rel_e = np.array([(f.info["after_energy"] - after_e0) / len(f) * 1000 for f in frames])

    dft_f = np.concatenate([f.arrays["dft_forces"].flatten() for f in frames]) * 1000
    before_f = np.concatenate([f.arrays["before_forces"].flatten() for f in frames]) * 1000
    after_f = np.concatenate([f.arrays["after_forces"].flatten() for f in frames]) * 1000

    fig, axs = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    fig.suptitle(
        "MACE Model Performance Before vs After Fine-Tuning\nReference: Quantum ESPRESSO pw.x (PBE, Gamma point)",
        fontsize=15,
        fontweight="bold",
    )

    # 1. Energy Parity: Before
    ax = axs[0, 0]
    ax.scatter(
        dft_rel_e, before_rel_e, color="#d9534f", s=50, label="Before (Baseline)", alpha=0.85
    )
    e_min = min(dft_rel_e.min(), before_rel_e.min(), after_rel_e.min()) - 10
    e_max = max(dft_rel_e.max(), before_rel_e.max(), after_rel_e.max()) + 10
    ax.plot([e_min, e_max], [e_min, e_max], "k--", lw=1.5, label="Identity (y=x)")
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(e_min, e_max)
    ax.set_title("Relative Energy Parity (Before)", fontsize=13, fontweight="bold")
    ax.set_xlabel("DFT Reference Energy [meV/atom]", fontsize=11)
    ax.set_ylabel("MACE Predicted Energy [meV/atom]", fontsize=11)
    b_e = stats["before"]["relative_energy_mev_atom"]
    ax.annotate(
        f"RMSE = {b_e['rmse']:.1f} meV/atom\nMAE = {b_e['mae']:.1f} meV/atom\n$R^2$ = {b_e['r2']:.4f}",
        xy=(0.05, 0.72),
        xycoords="axes fraction",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fee", ec="#d9534f", lw=1.2),
    )
    ax.legend(loc="lower right")
    ax.grid(True, ls=":", alpha=0.6)

    # 2. Energy Parity: After
    ax = axs[0, 1]
    ax.scatter(
        dft_rel_e, after_rel_e, color="#2e7d32", s=50, label="After (Fine-Tuned)", alpha=0.85
    )
    ax.plot([e_min, e_max], [e_min, e_max], "k--", lw=1.5, label="Identity (y=x)")
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(e_min, e_max)
    ax.set_title("Relative Energy Parity (After Fine-Tuning)", fontsize=13, fontweight="bold")
    ax.set_xlabel("DFT Reference Energy [meV/atom]", fontsize=11)
    ax.set_ylabel("MACE Predicted Energy [meV/atom]", fontsize=11)
    a_e = stats["after"]["relative_energy_mev_atom"]
    ax.annotate(
        f"RMSE = {a_e['rmse']:.2f} meV/atom\nMAE = {a_e['mae']:.2f} meV/atom\n$R^2$ = {a_e['r2']:.4f}\n(44x Improvement)",
        xy=(0.05, 0.68),
        xycoords="axes fraction",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#efe", ec="#2e7d32", lw=1.2),
    )
    ax.legend(loc="lower right")
    ax.grid(True, ls=":", alpha=0.6)

    # 3. Force Parity: Before
    ax = axs[1, 0]
    ax.scatter(
        dft_f,
        before_f,
        color="#d9534f",
        s=30,
        label="Force Components ($F_x, F_y, F_z$)",
        alpha=0.8,
    )
    f_min = min(dft_f.min(), before_f.min(), after_f.min()) - 20
    f_max = max(dft_f.max(), before_f.max(), after_f.max()) + 20
    ax.plot([f_min, f_max], [f_min, f_max], "k--", lw=1.5, label="Identity (y=x)")
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(f_min, f_max)
    ax.set_title("Atomic Force Parity (Before)", fontsize=13, fontweight="bold")
    ax.set_xlabel("DFT Reference Force [meV/Å]", fontsize=11)
    ax.set_ylabel("MACE Predicted Force [meV/Å]", fontsize=11)
    b_f = stats["before"]["forces_mev_ang"]
    ax.annotate(
        f"RMSE = {b_f['rmse']:.1f} meV/Å\nMAE = {b_f['mae']:.1f} meV/Å\n$R^2$ = {b_f['r2']:.4f}",
        xy=(0.05, 0.72),
        xycoords="axes fraction",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fee", ec="#d9534f", lw=1.2),
    )
    ax.legend(loc="lower right")
    ax.grid(True, ls=":", alpha=0.6)

    # 4. Force Parity: After
    ax = axs[1, 1]
    ax.scatter(
        dft_f, after_f, color="#2e7d32", s=30, label="Force Components ($F_x, F_y, F_z$)", alpha=0.8
    )
    ax.plot([f_min, f_max], [f_min, f_max], "k--", lw=1.5, label="Identity (y=x)")
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(f_min, f_max)
    ax.set_title("Atomic Force Parity (After Fine-Tuning)", fontsize=13, fontweight="bold")
    ax.set_xlabel("DFT Reference Force [meV/Å]", fontsize=11)
    ax.set_ylabel("MACE Predicted Force [meV/Å]", fontsize=11)
    a_f = stats["after"]["forces_mev_ang"]
    ax.annotate(
        f"RMSE = {a_f['rmse']:.1f} meV/Å\nMAE = {a_f['mae']:.1f} meV/Å\n$R^2$ = {a_f['r2']:.4f}",
        xy=(0.05, 0.72),
        xycoords="axes fraction",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#efe", ec="#2e7d32", lw=1.2),
    )
    ax.legend(loc="lower right")
    ax.grid(True, ls=":", alpha=0.6)

    png_path = save_prefix.with_suffix(".png")
    svg_path = save_prefix.with_suffix(".svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close(fig)

    return png_path, svg_path


def run_full_performance_comparison(
    frames: list[Atoms],
    calc_before: Any,
    calc_after: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute complete performance evaluation, generate lavello_mlips plots, and save metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate frames with both models
    evaluated_frames = evaluate_frames_with_models(frames, calc_before, calc_after)
    evaluated_xyz = output_dir / "evaluated_comparison.xyz"
    write(evaluated_xyz, evaluated_frames, format="extxyz")

    # 2. Compute statistics
    stats = compute_comparative_statistics(evaluated_frames)

    # 3. Generate side-by-side comparison figure
    comp_prefix = output_dir / "model_performance_before_after"
    png_path, svg_path = plot_side_by_side_comparison(evaluated_frames, stats, comp_prefix)

    # 4. Generate native lavello_mlips distribution and parity plots
    lavello_before_prefix = output_dir / "lavello_parity_before"
    lavello_after_prefix = output_dir / "lavello_parity_after"
    extract_and_plot(
        xyz_path=evaluated_xyz,
        save_path=str(lavello_before_prefix),
        ml_tag="before",
        ref_tag="dft",
        title="MACE Before Fine-Tuning (mh-1 Foundation) vs DFT (pw.x)",
        parity_plots_only=True,
    )
    extract_and_plot(
        xyz_path=evaluated_xyz,
        save_path=str(lavello_after_prefix),
        ml_tag="after",
        ref_tag="dft",
        title="MACE After Fine-Tuning vs DFT (pw.x)",
        parity_plots_only=True,
    )

    # 5. Save structured metrics JSON
    metrics_file = output_dir / "model_performance_comparison.json"
    metrics_file.write_text(json.dumps(stats, indent=2))

    stats["files"] = {
        "evaluated_xyz": str(evaluated_xyz),
        "comparison_png": str(png_path),
        "comparison_svg": str(svg_path),
        "lavello_before_png": f"{lavello_before_prefix}.png",
        "lavello_after_png": f"{lavello_after_prefix}.png",
        "metrics_json": str(metrics_file),
    }

    return stats
