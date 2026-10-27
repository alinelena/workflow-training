---
name: aiida-mlip-core
description: >-
  Core guide for STFC aiida-mlip: AiiDA profile setup, daemon management, code registration (janus@localhost),
  custom data types (ModelData, JanusConfigfile), and provenance queries.
  Use this skill when configuring AiiDA environments for MLIP calculations or initializing workflows.
---

# AiiDA-MLIP: Foundation & Configuration

`aiida-mlip` is an AiiDA plugin developed by STFC that connects machine-learned interatomic potentials (via `janus-core`) with automated data provenance, high-throughput calculation graphs, and distributed HPC scheduling.

## Core Capabilities

- **Calculations Supported**:
  - `mlip.sp`: Single-point energy, forces, Cauchy stress, and Hessians.
  - `mlip.opt`: Geometry optimization and crystal cell relaxation.
  - `mlip.md`: Molecular dynamics (NVE, NVT, NPT).
  - `mlip.ph`: Phonon dispersion, DOS, and vibrational thermodynamics.
  - `mlip.descriptors`: MLIP invariant representation extraction.
  - `mlip.train`: MACE model training and domain fine-tuning.
- **Custom Data Types**:
  - `ModelData` (`mlip.modeldata`): Tracks MLIP model checkpoints and weights with provenance.
  - `JanusConfigfile` (`mlip.config`): Stores declarative YAML simulation configurations as AiiDA database nodes.
- **High-Throughput Workflows**: High-throughput parameter sweeps and graph workflows via `aiida-workgraph`.

---

## Execution Environment & Rules

1. **Python Environment**: Run in the `janus` environment (`micromamba run -n janus python` or `micromamba run -n janus verdi ...`) or via `uv run --no-project python`.
2. **Daemon Status**: Verify the daemon is active (`verdi daemon status`) before submitting batch calculations.
3. **Headless Execution**: Pass `tracker: false` in YAML configs to avoid background CodeCarbon emissions overhead.
4. **Links**: Reference the [upstream GitHub repository](https://github.com/stfc/aiida-mlip) and [official documentation](https://stfc.github.io/aiida-mlip/).

---

## Getting Started

### 1. Profile Setup

Initialize an AiiDA profile with built-in storage:
```bash
micromamba run -n janus verdi presto
```

Verify profile and daemon:
```bash
micromamba run -n janus verdi status
micromamba run -n janus verdi daemon start
```

### 2. Registering the `janus` Code

Register `janus-core` CLI as an installed code in AiiDA (e.g. `janus@localhost`):
```bash
micromamba run -n janus verdi code create core.code.installed \
  --label janus \
  --computer localhost \
  --filepath-executable "$(which janus)"
```

Confirm available calculations:
```bash
micromamba run -n janus verdi plugin list aiida.calculations
```

### 3. AiiDA Data Types: `ModelData` and `JanusConfigfile`

Load or store an MLIP model in the AiiDA provenance graph:
```python
from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData

# Register model checkpoint
model = ModelData.from_local(
    "models/mace_mp_small.model",
    architecture="mace_mp",
)

# Register YAML configuration file
config = JanusConfigfile("config.yml")
```

### 4. Updating Skills from Upstream

Keep skills and reference docs synchronized with upstream GitHub repo (`stfc/aiida-mlip`) and docs:
```bash
./aiida-mlip/update.sh
# Check mode without writing changes:
./aiida-mlip/update.sh --check
```

See [entry_points.md](https://github.com/alinelena/ai-rules/blob/main/aiida-mlip/skills/aiida-mlip-core/references/entry_points.md) and [calculations_overview.md](https://github.com/alinelena/ai-rules/blob/main/aiida-mlip/skills/aiida-mlip-core/references/calculations_overview.md) for full reference tables.

