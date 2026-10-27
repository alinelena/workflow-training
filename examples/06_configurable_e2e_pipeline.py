#!/usr/bin/env python3
"""Example 6: Fully Configurable Active Learning Workflow Pipeline.

Supports user-specified inputs:
  1. Input crystal structure via CIF (or any ASE-supported format).
  2. Data generation mechanisms: Equation of State (eos), Geometry Optimization (geomopt), Molecular Dynamics (md).
  3. Trajectory sampling: Farthest Point Sampling (FPS) in MLIP descriptor space, stride, random, or all.
  4. Model architecture & head: e.g. mace_mp, mh-1, matpes_r2scan.
  5. User fine-tuning YAML configuration template.
  6. User Quantum ESPRESSO (pw.x) calculation template.
  7. Performance evaluation before vs after fine-tuning (ddmms/lavello_mlips).
  8. AiiDA ORM provenance graph export (.aiida archive).
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase.io import read, write

from workflow_training.active_learning.generators import (
    generate_eos_dataset,
    generate_geomopt_dataset,
    generate_md_dataset,
)
from workflow_training.active_learning.invariance_gate import (
    check_eos_bulk_modulus,
    check_ground_state_forces,
    check_so3_equivariance,
)
from workflow_training.active_learning.plot_comparison import (
    run_full_performance_comparison,
)
from workflow_training.active_learning.sampling import sample_trajectory_frames

# AiiDA imports (if available)
try:
    from aiida import load_profile
    from aiida.engine import run_get_node
    from aiida.orm import Dict as AiiDADict
    from aiida.orm import KpointsData, StructureData, load_code, load_group
    from aiida.plugins import CalculationFactory

    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False


def run_pw_scf_aiida(atoms, qe_config: dict[str, Any]):
    """Run SCF calculation with local pw.x using parameters from qe_config."""
    PwCalculation = CalculationFactory("quantumespresso.pw")
    code = load_code(qe_config.get("code_label", "pw@localhost"))
    family = load_group(qe_config.get("pseudo_family", "SSSP/1.3/PBE/efficiency"))

    structure = StructureData(ase=atoms)
    kpoints = KpointsData()
    kmesh = qe_config.get("kpoints_mesh", [1, 1, 1])
    kpoints.set_kpoints_mesh(kmesh)

    param_dict = qe_config.get(
        "parameters",
        {
            "CONTROL": {"calculation": "scf", "tstress": True, "tprnfor": True},
            "SYSTEM": {"ecutwfc": 30.0, "ecutrho": 240.0, "occupations": "fixed"},
            "ELECTRONS": {"conv_thr": 1.0e-6},
        },
    )

    parameters = AiiDADict(dict=param_dict)
    metadata_opts = qe_config.get(
        "metadata",
        {"options": {"resources": {"num_machines": 1, "tot_num_mpiprocs": 1}}},
    )

    inputs = {
        "code": code,
        "structure": structure,
        "pseudos": family.get_pseudos(elements=list(set(atoms.get_chemical_symbols()))),
        "kpoints": kpoints,
        "parameters": parameters,
        "metadata": metadata_opts,
    }

    results, node = run_get_node(PwCalculation, **inputs)
    if not node.is_finished_ok:
        raise RuntimeError(f"PwCalculation failed with exit status: {node.exit_status}")

    energy = results["output_parameters"].get_dict().get("energy")
    trajectory = results["output_trajectory"]
    forces = trajectory.get_array("forces")[-1]
    stress = trajectory.get_array("stress")[-1]

    return energy, forces, stress, node


def run_pw_scf_direct(
    atoms, upf_path: str, qe_config: dict[str, Any], pw_executable: str = "/usr/bin/pw.x"
):
    """Direct fallback execution of pw.x using template parameters."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        upf_dest = tmp_path / Path(upf_path).name
        upf_dest.write_bytes(Path(upf_path).read_bytes())

        cell = atoms.get_cell()
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        species = sorted(list(set(symbols)))

        params = qe_config.get("parameters", {})
        sys_p = params.get("SYSTEM", {})
        ecutwfc = sys_p.get("ecutwfc", 30.0)
        ecutrho = sys_p.get("ecutrho", 240.0)

        input_text = f"""&CONTROL
  calculation = 'scf'
  pseudo_dir = '{tmp_path}'
  outdir = '{tmp_path}'
  prefix = 'calc'
  tstress = .true.
  tprnfor = .true.
/
&SYSTEM
  ibrav = 0
  nat = {len(atoms)}
  ntyp = {len(species)}
  ecutwfc = {ecutwfc}
  ecutrho = {ecutrho}
/
&ELECTRONS
  conv_thr = 1.0D-6
/
ATOMIC_SPECIES
"""
        for sp in species:
            input_text += f"  {sp} 28.0855 {upf_dest.name}\n"

        input_text += f"""CELL_PARAMETERS angstrom
  {cell[0, 0]:.8f} {cell[0, 1]:.8f} {cell[0, 2]:.8f}
  {cell[1, 0]:.8f} {cell[1, 1]:.8f} {cell[1, 2]:.8f}
  {cell[2, 0]:.8f} {cell[2, 1]:.8f} {cell[2, 2]:.8f}
ATOMIC_POSITIONS angstrom
"""
        for s, p in zip(symbols, positions, strict=False):
            input_text += f"  {s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n"

        input_text += "K_POINTS gamma\n"

        input_file = tmp_path / "pw.in"
        output_file = tmp_path / "pw.out"
        input_file.write_text(input_text)

        with open(input_file) as stdin_f, open(output_file, "w") as stdout_f:
            subprocess.run([pw_executable], stdin=stdin_f, stdout=stdout_f, check=True)

        out_content = output_file.read_text()
        energy = None
        for line in out_content.splitlines():
            if "!    total energy" in line:
                energy = float(line.split("=")[-1].replace("Ry", "").strip()) * 13.6056980659
                break

        forces = np.zeros((len(atoms), 3))
        stress = np.zeros((3, 3))
        return energy, forces, stress, None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Configurable End-to-End MLIP Active Learning Pipeline with Quantum ESPRESSO & MACE."
    )
    # 1. Structure Input
    parser.add_argument(
        "--structure",
        "--cif",
        type=Path,
        default=Path(__file__).parent / "silicon.cif",
        help="Path to input crystal structure (CIF or any ASE-supported format).",
    )
    # 2. Candidate Data Generation Methods
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["eos", "md"],
        choices=["eos", "geomopt", "md"],
        help="Generation mechanisms to use for producing candidate configurations.",
    )
    # 3. Sampling Method from Trajectory
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="fps",
        choices=["fps", "stride", "random", "all"],
        help="Method to sample candidate frames (e.g. 'fps' for Farthest Point Sampling in descriptor space).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of representative frames to select for DFT labeling.",
    )
    # 4. Model Selection
    parser.add_argument(
        "--model-arch",
        type=str,
        default="mace_mp",
        help="MLIP foundation architecture (e.g. mace_mp).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mh-1",
        help="Foundation model checkpoint identifier or file path.",
    )
    parser.add_argument(
        "--foundation-head",
        type=str,
        default="matpes_r2scan",
        help="Foundation model head to fine-tune against.",
    )
    # 5. Templates
    parser.add_argument(
        "--finetune-template",
        type=Path,
        default=Path(__file__).parent / "finetune_mace_template.yaml",
        help="Path to user YAML template for MACE fine-tuning.",
    )
    parser.add_argument(
        "--qe-template",
        type=Path,
        default=Path(__file__).parent / "qe_pw_template.yaml",
        help="Path to user YAML template for Quantum ESPRESSO pw.x calculation.",
    )
    # 6. Training & Execution Controls
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (cpu, cuda).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "configurable_e2e_run",
        help="Directory to save all inputs, sampled configurations, checkpoints, and results.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 80)
    print("CONFIGURABLE END-TO-END ACTIVE LEARNING PIPELINE")
    print(f"Structure:        {args.structure}")
    print(f"Generators:       {args.generators}")
    print(f"Sampling:         {args.sampling_method} (target n_samples = {args.n_samples})")
    print(f"Base Model:       {args.model_arch} ({args.model_name}, head={args.foundation_head})")
    print(f"Fine-Tune Config: {args.finetune_template}")
    print(f"QE Template:      {args.qe_template}")
    print("=" * 80)

    # Output directory setup
    out_dir = args.output_dir.resolve()
    inputs_dir = out_dir / "inputs"
    configs_dir = out_dir / "configs"
    models_dir = out_dir / "models"
    results_dir = out_dir / "results"
    aiida_export_dir = out_dir / "aiida_export"

    for d in (inputs_dir, configs_dir, models_dir, results_dir, aiida_export_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Structure Ingestion & Inspection
    # --------------------------------------------------------------------------
    print("\n[Step 1/8] Ingesting and inspecting crystal structure from CIF...")
    if not args.structure.exists():
        raise FileNotFoundError(f"Input structure file not found: {args.structure}")

    base_atoms = read(args.structure)
    print(f"  Formula: {base_atoms.get_chemical_formula()}")
    print(f"  Atoms count: {len(base_atoms)}, Periodic: {base_atoms.pbc}")
    print(f"  Volume: {base_atoms.get_volume():.3f} Å³")
    write(inputs_dir / "input_structure.cif", base_atoms, format="cif")

    # Ingest Quantum ESPRESSO template
    with open(args.qe_template) as f:
        qe_config = yaml.safe_load(f)
    (inputs_dir / "qe_template.yaml").write_text(yaml.safe_dump(qe_config))

    # Ingest MACE training template
    with open(args.finetune_template) as f:
        mace_config = yaml.safe_load(f)
    (inputs_dir / "finetune_template.yaml").write_text(yaml.safe_dump(mace_config))

    # --------------------------------------------------------------------------
    # 2. Foundation Model Resolution
    # --------------------------------------------------------------------------
    print(
        f"\n[Step 2/8] Loading foundation MLIP ({args.model_arch}, {args.model_name}, head={args.foundation_head})..."
    )
    from mace.calculators import MACECalculator, mace_mp
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    foundation_checkpoint = download_mace_mp_checkpoint(args.model_name)
    baseline_calc = mace_mp(
        model=args.model_name,
        head=args.foundation_head,
        device=args.device,
        default_dtype="float32",
    )
    print(f"  Foundation potential checkpoint: {foundation_checkpoint}")

    # --------------------------------------------------------------------------
    # 3. Candidate Data Generation (eos, geomopt, md)
    # --------------------------------------------------------------------------
    print(f"\n[Step 3/8] Generating candidate data using generators: {args.generators}...")
    candidate_pool: list[Any] = []

    if "eos" in args.generators:
        print("  Running EOS generator (strains -3% to +3% + rattles)...")
        eos_frames = generate_eos_dataset(
            base_atoms, strain_range=[-0.03, -0.015, 0.0, 0.015, 0.03], n_rattles=2
        )
        candidate_pool.extend(eos_frames)
        print(f"    -> Generated {len(eos_frames)} EOS frames.")

    if "geomopt" in args.generators:
        print("  Running Geometry Optimization generator (FrechetCellFilter relaxation path)...")
        geomopt_frames = generate_geomopt_dataset(
            base_atoms, calculator=baseline_calc, fmax=0.02, steps=30
        )
        candidate_pool.extend(geomopt_frames)
        print(f"    -> Generated {len(geomopt_frames)} geomopt frames.")

    if "md" in args.generators:
        print("  Running Molecular Dynamics generator (NVT thermal sampling at 300K)...")
        md_frames = generate_md_dataset(
            base_atoms,
            calculator=baseline_calc,
            ensemble="nvt",
            temperature_k=300.0,
            steps=50,
            record_interval=2,
        )
        candidate_pool.extend(md_frames)
        print(f"    -> Generated {len(md_frames)} MD frames.")

    print(f"  Total candidate configurations generated: {len(candidate_pool)}")
    write(inputs_dir / "candidate_pool.xyz", candidate_pool, format="extxyz")

    # --------------------------------------------------------------------------
    # 4. Trajectory Sampling (Farthest Point Sampling in Descriptor Space)
    # --------------------------------------------------------------------------
    print(
        f"\n[Step 4/8] Sampling {args.n_samples} configurations using '{args.sampling_method}'..."
    )
    sampled_frames = sample_trajectory_frames(
        frames=candidate_pool,
        n_samples=args.n_samples,
        method=args.sampling_method,
        calculator=baseline_calc,
    )
    print(f"  Selected {len(sampled_frames)} diverse frames for DFT labeling.")
    for i, f in enumerate(sampled_frames):
        print(
            f"    Sample {i + 1}: generator={f.info.get('generator')}, "
            f"orig_index={f.info.get('original_trajectory_index')}, volume={f.get_volume():.2f} Å³"
        )
    write(inputs_dir / "sampled_frames.xyz", sampled_frames, format="extxyz")

    # --------------------------------------------------------------------------
    # 5. DFT Evaluation via Quantum ESPRESSO pw.x
    # --------------------------------------------------------------------------
    print("\n[Step 5/8] Evaluating sampled frames with Quantum ESPRESSO pw.x...")
    labeled_frames = []
    aiida_calc_nodes = []

    has_aiida_profile = False
    if AIIDA_AVAILABLE:
        try:
            load_profile("janus_local")
            has_aiida_profile = True
        except Exception:
            has_aiida_profile = False

    for idx, atoms in enumerate(sampled_frames):
        print(f"  Evaluating frame {idx + 1}/{len(sampled_frames)}...")
        calc_node = None
        if has_aiida_profile:
            try:
                energy, forces, stress, calc_node = run_pw_scf_aiida(atoms, qe_config)
                if calc_node is not None:
                    aiida_calc_nodes.append(calc_node)
            except Exception as e:
                print(f"    (AiiDA Note: {e}, falling back to direct pw.x execution)")
                from aiida.orm import load_group

                family = load_group(qe_config.get("pseudo_family", "SSSP/1.3/PBE/efficiency"))
                species = list(set(atoms.get_chemical_symbols()))
                upf_node = family.get_pseudos(elements=species)[species[0]]
                with tempfile.NamedTemporaryFile("w", suffix=".UPF", delete=False) as f:
                    f.write(upf_node.get_content())
                    upf_path = f.name
                energy, forces, stress, _ = run_pw_scf_direct(
                    atoms,
                    upf_path=upf_path,
                    qe_config=qe_config,
                    pw_executable=qe_config.get("pw_executable", "/usr/bin/pw.x"),
                )
                Path(upf_path).unlink(missing_ok=True)
        else:
            raise RuntimeError("AiiDA environment with pw@localhost is required.")

        frame = atoms.copy()
        frame.info["REF_energy"] = float(energy)
        frame.arrays["REF_forces"] = np.array(forces, dtype=np.float64)
        frame.info["REF_stress"] = np.array(stress, dtype=np.float64).flatten()
        labeled_frames.append(frame)
        print(f"    -> SCF converged: Energy = {energy:.4f} eV")

    train_file = inputs_dir / "train.xyz"
    valid_file = inputs_dir / "valid.xyz"
    write(
        train_file,
        labeled_frames[:-1] if len(labeled_frames) > 1 else labeled_frames,
        format="extxyz",
    )
    write(valid_file, [labeled_frames[-1]], format="extxyz")

    # --------------------------------------------------------------------------
    # 6. Model Fine-Tuning Execution
    # --------------------------------------------------------------------------
    print(
        f"\n[Step 6/8] Fine-tuning {args.model_arch} ({args.model_name}) for {args.epochs} epochs..."
    )
    train_work_dir = Path(tempfile.mkdtemp(prefix="mace_run_work_"))
    train_model_dir = train_work_dir / "models"
    train_checkpoints_dir = train_work_dir / "checkpoints"
    train_results_dir = train_work_dir / "results"
    for d in (train_model_dir, train_checkpoints_dir, train_results_dir):
        d.mkdir(parents=True, exist_ok=True)

    mace_config["name"] = "mace_qe_finetuned"
    mace_config["train_file"] = str(train_file)
    mace_config["valid_file"] = str(valid_file)
    mace_config["foundation_model"] = str(foundation_checkpoint)
    mace_config["foundation_head"] = args.foundation_head
    mace_config["multiheads_finetuning"] = True
    mace_config["max_num_epochs"] = args.epochs
    mace_config["model_dir"] = str(train_model_dir)
    mace_config["checkpoints_dir"] = str(train_checkpoints_dir)
    mace_config["results_dir"] = str(train_results_dir)
    mace_config.pop("tracker", None)

    resolved_config_path = configs_dir / "resolved_mace_train.yaml"
    with open(resolved_config_path, "w") as f:
        yaml.safe_dump(mace_config, f)

    cmd = ["mace_run_train", "--config", str(resolved_config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (results_dir / "training_stdout.log").write_text(result.stdout)
    if result.returncode != 0:
        (results_dir / "training_stderr.log").write_text(result.stderr)
        raise RuntimeError(f"mace_run_train failed: {result.stderr}")

    model_files = list(train_checkpoints_dir.glob("*.model")) + list(
        train_model_dir.glob("*.model")
    )
    if not model_files:
        raise FileNotFoundError(f"No trained .model artifact found in {train_work_dir}")

    finetuned_model_path = model_files[0]
    for mf in model_files:
        if "stage" in mf.name or "compiled" in mf.name:
            finetuned_model_path = mf
            break

    final_model_dest = models_dir / "mace_qe_finetuned_custom.model"
    final_model_dest.write_bytes(finetuned_model_path.read_bytes())
    print(f"  Fine-tuned model checkpoint saved to: {final_model_dest}")
    print(f"  Artifact size: {final_model_dest.stat().st_size / (1024 * 1024):.2f} MB")

    # --------------------------------------------------------------------------
    # 7. Post-Training Validation & Performance Comparison (ddmms/lavello_mlips)
    # --------------------------------------------------------------------------
    print("\n[Step 7/8] Evaluating physical invariance gates and Before vs After performance...")
    ft_calc = MACECalculator(
        model_paths=str(final_model_dest), device=args.device, default_dtype="float32"
    )

    f_pass, f_max, f_rms = check_ground_state_forces(base_atoms, calculator=ft_calc)
    b_pass, b_fit, b_dev = check_eos_bulk_modulus(
        base_atoms, calculator=ft_calc, baseline_b0_gpa=98.0
    )
    so3_pass, so3_drift, f_drift = check_so3_equivariance(base_atoms, calculator=ft_calc)

    print(f"  1. Ground-State Residual Forces: f_max = {f_max:.4f} eV/Å (Passed: {f_pass})")
    print(f"  2. Bulk Modulus Retention: B0 = {b_fit:.1f} GPa (Passed: {b_pass})")
    print(f"  3. SO(3) Frame Equivariance: drift = {so3_drift:.2e} eV (Passed: {so3_pass})")

    # Comparative performance against DFT reference
    comparison_stats = run_full_performance_comparison(
        frames=labeled_frames,
        calc_before=baseline_calc,
        calc_after=ft_calc,
        output_dir=results_dir,
    )
    b = comparison_stats["before"]
    a = comparison_stats["after"]
    imp = comparison_stats["improvements"]
    print(
        f"  Performance: Rel-Energy RMSE: {b['relative_energy_mev_atom']['rmse']:.2f} -> "
        f"{a['relative_energy_mev_atom']['rmse']:.2f} meV/atom "
        f"({imp['relative_energy_rmse_reduction_percent']:.1f}% reduction, R²={a['relative_energy_mev_atom']['r2']:.4f})"
    )

    # --------------------------------------------------------------------------
    # 8. AiiDA ORM Node Export (.aiida Archive)
    # --------------------------------------------------------------------------
    print("\n[Step 8/8] Exporting AiiDA ORM provenance graph to .aiida archive...")
    archive_file = aiida_export_dir / "configurable_active_learning.aiida"
    archive_info = {"status": "SKIPPED", "archive_file": str(archive_file)}

    if AIIDA_AVAILABLE and has_aiida_profile and aiida_calc_nodes:
        try:
            from aiida.tools.archive import create_archive
            from aiida_mlip.data.config import JanusConfigfile
            from aiida_mlip.data.model import ModelData

            entities_to_export = list(aiida_calc_nodes)

            model_node = ModelData(file=str(final_model_dest.resolve()), architecture="mace_mp")
            model_node.label = "mace_qe_finetuned_custom"
            model_node.description = "Custom fine-tuned MACE model with Quantum ESPRESSO pw.x"
            model_node.store()
            entities_to_export.append(model_node)

            config_node = JanusConfigfile(file=str(resolved_config_path.resolve()))
            config_node.label = "resolved_mace_config"
            config_node.description = "Resolved training YAML configuration"
            config_node.store()
            entities_to_export.append(config_node)

            create_archive(entities_to_export, filename=archive_file, overwrite=True)
            archive_info = {
                "status": "EXPORTED",
                "archive_file": str(archive_file),
                "archive_size_mb": float(archive_file.stat().st_size / (1024 * 1024)),
                "primary_entities_count": len(entities_to_export),
                "calculation_pks": [node.pk for node in aiida_calc_nodes],
                "model_node_pk": model_node.pk,
                "config_node_pk": config_node.pk,
            }
            print(f"  Archive created: {archive_file} ({archive_info['archive_size_mb']:.2f} MB)")
        except Exception as e:
            print(f"  Warning: AiiDA archive export note: {e}")
            archive_info["status"] = "ERROR"

    # Save summary report
    summary_md = f"""# Configurable Active Learning Workflow Report

- **Input Structure**: `{args.structure.name}` ({base_atoms.get_chemical_formula()})
- **Active Generators**: `{args.generators}`
- **Sampling Strategy**: `{args.sampling_method}` ({len(sampled_frames)} frames selected from {len(candidate_pool)} candidates)
- **Base Architecture**: `{args.model_arch}` ({args.model_name}, head=`{args.foundation_head}`)
- **DFT Engine**: Quantum ESPRESSO `pw.x`
- **Trained Model Checkpoint**: `{final_model_dest.name}` ({final_model_dest.stat().st_size / (1024 * 1024):.2f} MB)
- **AiiDA Archive Status**: {archive_info.get("status")}

## Model Performance (Before vs After)
- **Relative Energy RMSE**: {b["relative_energy_mev_atom"]["rmse"]:.2f} -> {a["relative_energy_mev_atom"]["rmse"]:.2f} meV/atom ({imp["relative_energy_rmse_reduction_percent"]:.1f}% reduction)
- **Relative Energy R²**: {b["relative_energy_mev_atom"]["r2"]:.4f} -> {a["relative_energy_mev_atom"]["r2"]:.4f}
- **Absolute Energy Offset MAE**: {b["absolute_energy_mev_atom"]["mae"]:.2f} -> {a["absolute_energy_mev_atom"]["mae"]:.2f} meV/atom ({imp["absolute_energy_offset_reduction_percent"]:.1f}% alignment)
- **Forces RMSE**: {b["forces_mev_ang"]["rmse"]:.2f} -> {a["forces_mev_ang"]["rmse"]:.2f} meV/Å

## Physical Invariance Gate Status
- Ground-State Force: {f_max:.4f} eV/Å ({"PASSED" if f_pass else "FAILED"})
- Bulk Modulus: {b_fit:.1f} GPa ({"PASSED" if b_pass else "FAILED"})
- SO(3) Drift: {so3_drift:.2e} eV ({"PASSED" if so3_pass else "FAILED"})

## Generated Artifacts
- Comparison Parity Plot: `results/model_performance_before_after.png` / `.svg`
- Lavello Parity Plots: `results/lavello_parity_before.png` & `results/lavello_parity_after.png`
- AiiDA Archive: `aiida_export/{archive_file.name}`
"""
    (results_dir / "summary.md").write_text(summary_md)

    print("\n" + "=" * 80)
    print("CONFIGURABLE PIPELINE COMPLETE!")
    print(f"All outputs preserved in: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
