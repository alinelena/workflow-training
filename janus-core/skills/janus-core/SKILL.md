---
name: janus-core
description: >-
  Core workflow guide for STFC janus-core: MLIP architecture selection (MACE, SevenNet, CHGNet, M3GNet, etc.),
  device management (CPU, CUDA, MPS), configuration file workflows, and Context7 live documentation lookup.
  Use this skill when starting any atomistic simulation or configuring MLIP calculators.
---

# Janus-Core: Foundation & MLIP Management

`janus-core` is an open-source Python library and CLI tool developed by the UKRI Science and Technology Facilities Council (STFC) for atomistic calculations with machine-learned interatomic potentials (MLIPs).

## Core Capabilities

- **Unified Calculator Interface**: Supports state-of-the-art MLIPs:
  - `mace_mp` / `mace_off` / `mace` / `mace_omol` / `mace_polar`: Materials and molecular foundation potentials.
  - `sevennet`: High-accuracy graph neural network potential for inorganic systems.
  - `chgnet`: Crystal Hamiltonian Graph Neural Network (oxides, batteries).
  - `nequip`: E(3)-equivariant interatomic potential for niche high-precision systems.
  - `deepmd` (`dpa3`): Deep Potential architecture for condensed matter at scale.
  - `orb`: Orbital Materials foundation potential.
  - `mattersim`: Microsoft universal materials potential.
  - `grace`: Multi-element tensor potential for alloys and crystals.
  - `upet`: Universal Pretrained Equivariant Transformer.
  - `fairchem`: Open Catalyst and FAIRChem models.
- **Hardware Acceleration**: Automatic GPU (`cuda`), Apple Silicon (`mps`), Intel GPU (`xpu`), and CPU (`cpu`) acceleration.
- **YAML Driven**: All CLI commands accept structured YAML configuration files via `--config <path>`.
- **Live Documentation**: Official docs at `https://stfc.github.io/janus-core/` and Context7 live lookup at `https://context7.com/stfc/janus-core`.

---

## Multi-Agent Execution Rules

1. **Python Environment**: Run in the `janus` micromamba environment (`micromamba run -n janus python`) or via `uv run --no-project python`.
2. **Headless & CI Execution**: Always pass `--no-tracker` in automated scripts to disable CodeCarbon emission tracking and background pynvml warnings.
3. **Structure Verification**: Always inspect structure coordinates, cell parameters, and periodic boundary conditions using `janus-core/tools/structure_inspector.py <structure>` before executing heavy simulations.
4. **Cell Relaxation Filter**: For crystal geometry optimizations with cell changes, always prefer `--filter FrechetCellFilter` (or `filter_func="FrechetCellFilter"`).

---

## Quick Start Procedures

### 1. Verify Atomic Structure

Inspect atomic composition, cell parameters, and PBC before simulation:
```bash
python janus-core/tools/structure_inspector.py structure.cif
# or output JSON for automated parsing:
python janus-core/tools/structure_inspector.py structure.cif --json
```

### 2. Using YAML Configuration Files

All `janus` subcommands accept a `--config` YAML file containing options:
```bash
janus geomopt --config geomopt_config.yml
```

Example YAML configuration (`janus_config.yml`):
```yaml
arch: mace_mp
device: cpu
struct: structure.cif
file_prefix: results/NaCl
tracker: false
```

### 3. Choosing the MLIP Architecture

Pass `--arch <arch>` and optionally `--model <model-path-or-id>`:
- **Solids & Inorganics**: `--arch mace_mp` or `--arch sevennet` or `--arch chgnet`
- **Organic Molecules**: `--arch mace_off` or `--arch mace_omol`
- **Large-scale Condensed Matter**: `--arch deepmd`

Detailed architecture guide: see [architectures.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-core/references/architectures.md).

### 4. Updating Janus-Core Skills from Upstream

Keep skills and reference docs synchronized with upstream GitHub repo (`stfc/janus-core`) and docs (`stfc.github.io/janus-core`):
```bash
./janus-core/update.sh
# Check mode without writing changes:
./janus-core/update.sh --check
```

