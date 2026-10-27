#!/usr/bin/env python3
"""Plot Training Results & Model Performance Comparison.

Inspired by and compatible with ddmms/lavello_mlips (plot_train_results.py).
Evaluates MACE potential performance Before vs After fine-tuning against
reference Quantum ESPRESSO pw.x calculations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read
from mace.calculators import MACECalculator, mace_mp

from workflow_training.active_learning.plot_comparison import (
    run_full_performance_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot MLIP model performance before and after fine-tuning."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=[
            Path("examples/mace_qe_finetune_run/inputs/train.xyz"),
            Path("examples/mace_qe_finetune_run/inputs/valid.xyz"),
        ],
        help="Path(s) to dataset XYZ file(s) containing DFT reference calculations.",
    )
    parser.add_argument(
        "--model-after",
        type=Path,
        default=Path("examples/mace_qe_finetune_run/models/mace_qe_finetuned_small.model"),
        help="Path to the fine-tuned MACE model checkpoint.",
    )
    parser.add_argument(
        "--foundation-head",
        type=str,
        default="matpes_r2scan",
        help="Foundation model head to evaluate for the baseline before fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/mace_qe_finetune_run/results"),
        help="Directory to save comparison plots and metrics JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for inference (cpu, cuda).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("MLIP Performance Comparison (Before vs After Fine-Tuning)")
    print("=" * 75)

    # 1. Load dataset frames
    frames = []
    for p in args.dataset:
        if p.exists():
            f = read(p, index=":")
            frames.extend(f)
            print(f"  Loaded {len(f)} frames from {p}")
        else:
            print(f"  Warning: File not found: {p}")

    if not frames:
        raise FileNotFoundError("No frames found to evaluate.")

    # 2. Instantiate calculators
    print(f"\nLoading baseline foundation model (mace_mp mh-1, head={args.foundation_head})...")
    calc_before = mace_mp(
        model="mh-1",
        head=args.foundation_head,
        device=args.device,
        default_dtype="float32",
    )

    print(f"Loading fine-tuned model checkpoint ({args.model_after})...")
    calc_after = MACECalculator(
        model_paths=str(args.model_after.resolve()),
        device=args.device,
        default_dtype="float32",
    )

    # 3. Run full comparison
    print(f"\nEvaluating {len(frames)} configurations and generating parity plots...")
    stats = run_full_performance_comparison(
        frames=frames,
        calc_before=calc_before,
        calc_after=calc_after,
        output_dir=args.output_dir,
    )

    b = stats["before"]
    a = stats["after"]
    imp = stats["improvements"]

    print("\n" + "-" * 75)
    print("COMPARATIVE PERFORMANCE SUMMARY:")
    print("-" * 75)
    print(
        f"Relative Energy RMSE: {b['relative_energy_mev_atom']['rmse']:.2f} -> {a['relative_energy_mev_atom']['rmse']:.2f} meV/atom "
        f"({imp['relative_energy_rmse_reduction_percent']:.1f}% reduction)"
    )
    print(
        f"Relative Energy MAE:  {b['relative_energy_mev_atom']['mae']:.2f} -> {a['relative_energy_mev_atom']['mae']:.2f} meV/atom"
    )
    print(
        f"Relative Energy R²:   {b['relative_energy_mev_atom']['r2']:.4f} -> {a['relative_energy_mev_atom']['r2']:.4f}"
    )
    print(
        f"Forces Component RMSE: {b['forces_mev_ang']['rmse']:.2f} -> {a['forces_mev_ang']['rmse']:.2f} meV/Å"
    )
    print(
        f"Absolute Energy Offset MAE: {b['absolute_energy_mev_atom']['mae']:.2f} -> {a['absolute_energy_mev_atom']['mae']:.2f} meV/atom "
        f"({imp['absolute_energy_offset_reduction_percent']:.1f}% alignment)"
    )
    print("-" * 75)
    print(f"Generated side-by-side plot: {stats['files']['comparison_png']}")
    print(f"Generated SVG graphic:       {stats['files']['comparison_svg']}")
    print(f"Generated lavello Before:    {stats['files']['lavello_before_png']}")
    print(f"Generated lavello After:     {stats['files']['lavello_after_png']}")
    print(f"Generated metrics JSON:      {stats['files']['metrics_json']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
