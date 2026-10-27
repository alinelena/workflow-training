# Janus-Core CLI Overview

The STFC `janus-core` CLI entrypoint is `janus`.

```text
Usage: janus [OPTIONS] COMMAND [ARGS]...

Calculations:
  singlepoint   Perform single point calculations and save to file.
  geomopt       Perform geometry optimization and save optimized structure.
  md            Run molecular dynamics simulation (NVE, NVT, NPT).
  phonons       Calculate phonons and save band structures / DOS.
  eos           Calculate equation of state and bulk modulus.
  elasticity    Calculate elasticity tensors.
  neb           Run Nudged Elastic Band method.
  descriptors   Calculate MLIP descriptors.

Training:
  train         Train or fine-tune an MLIP.
  preprocess    Preprocess data before training.
```

## Universal Flags Across All Commands

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config` | PATH | None | Path to YAML configuration file supplying CLI options. |
| `--arch` | STRING | mace_mp | Machine Learned Interatomic Potential architecture. |
| `--device` | STRING | cpu | Compute device (`cpu`, `cuda`, `mps`, `xpu`). |
| `--model` | PATH / STR | None | Path to custom checkpoint file or model identifier. |
| `--file-prefix`| PATH | ./janus_results | Directory and filename prefix for outputs. |
| `--no-tracker` | FLAG | (tracker=True) | Disable CodeCarbon emissions tracking. |
| `--log` | PATH | auto | Path to write execution log file. |
| `--summary` | PATH | auto | Path to write execution summary YAML. |
| `--help` | FLAG | None | Display subcommand options and help. |

## Subcommands Registered in CLI

- **`janus descriptors`**: Key options: `--arch`, `--struct`, `--invariants-only`, `--calc-per-element`, `--calc-per-atom`, `--out`, ...
- **`janus elasticity`**: Key options: `--arch`, `--struct`, `--shear-magnitude`, `--normal-magnitude`, `--n-strains`, `--write-voigt`, ...
- **`janus eos`**: Key options: `--arch`, `--struct`, `--min-volume`, `--max-volume`, `--n-volumes`, `--eos-type`, ...
- **`janus geomopt`**: Key options: `--arch`, `--struct`, `--optimizer`, `--fmax`, `--steps`, `--opt-cell-lengths`, ...
- **`janus md`**: Key options: `--arch`, `--ensemble`, `--struct`, `--steps`, `--timestep`, `--temp`, ...
- **`janus neb`**: Key options: `--arch`, `--init-struct`, `--final-struct`, `--neb-structs`, `--neb-class`, `--n-images`, ...
- **`janus phonons`**: Key options: `--arch`, `--struct`, `--supercell`, `--displacement`, `--displacement-kwargs`, `--mesh`, ...
- **`janus preprocess`**: Key options: `--mlip-config`, `--log`, `--tracker`, `--summary`
- **`janus singlepoint`**: Key options: `--arch`, `--struct`, `--properties`, `--out`, `--device`, `--model`, ...
- **`janus train`**: Key options: `--arch`, `--mlip-config`, `--fine-tune`, `--file-prefix`, `--log`, `--tracker`, ...
