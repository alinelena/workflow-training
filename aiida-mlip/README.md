# `aiida-mlip` — AI Skills & Rules Group for Antigravity

This directory contains a complete, self-contained capability group of **AI Skills**, **Execution Rules**, **Reference Documentation**, and **Workflow Examples** for STFC [`aiida-mlip`](https://github.com/stfc/aiida-mlip) (version 0.5.0+).

---

## What Is Included

### 1. Skills Group (`aiida-mlip/skills/`)

| Skill | Slash Command | Capabilities & Scope |
| :--- | :--- | :--- |
| **`aiida-mlip-core`** | `/aiida-mlip-core` | Foundation guide: AiiDA profile setup, code configuration (`janus@localhost`), `ModelData`, `JanusConfigfile`, and entry points. |
| **`aiida-mlip-singlepoint`** | `/aiida-mlip-singlepoint` | Static single-point energy, forces, Cauchy stress tensors, and Hessians via `mlip.sp`. |
| **`aiida-mlip-geomopt`** | `/aiida-mlip-geomopt` | Geometry optimization and cell relaxation (`FrechetCellFilter`, variable cell) via `mlip.opt`. |
| **`aiida-mlip-md`** | `/aiida-mlip-md` | Molecular dynamics simulations across NVE, NVT, NPT ensembles via `mlip.md`. |
| **`aiida-mlip-phonons`** | `/aiida-mlip-phonons` | Phonon band structures, DOS, and vibrational thermodynamics via `mlip.ph` and Phonopy. |
| **`aiida-mlip-descriptors`** | `/aiida-mlip-descriptors` | MLIP invariant representations extraction (mean, per-atom, per-element) via `mlip.descriptors`. |
| **`aiida-mlip-train`** | `/aiida-mlip-train` | MACE and MLIP model training and transfer learning fine-tuning via `mlip.train`. |
| **`aiida-mlip-workflows`** | `/aiida-mlip-workflows` | High-throughput workgraphs with `aiida-workgraph` (`build_ht_calc`) across structure directories. |

### 2. Multi-Agent Execution Rules (`aiida-mlip/rules/`)

- `aiida-mlip-rules.md` / `AGENTS.md`: Enforces key execution guidelines:
  - Run in the `janus` micromamba environment (`micromamba run -n janus python` or `micromamba run -n janus verdi ...`) or via `uv`.
  - Verify AiiDA daemon status (`verdi daemon status`) before submitting batch calculations.
  - Set `tracker: false` in YAML configuration files to prevent unnecessary emissions overhead.
  - Ensure `janus@localhost` code is properly configured.
  - Official documentation references ([aiida-mlip Docs](https://stfc.github.io/aiida-mlip/)).

### 3. Update Mechanism (`aiida-mlip/update.sh` and `aiida-mlip/scripts/update_skills.py`)

A zero-dependency update mechanism that keeps all skills and reference documentation synchronized with:
- **Upstream GitHub Repository**: [`https://github.com/stfc/aiida-mlip`](https://github.com/stfc/aiida-mlip)
- **Official Documentation**: [`https://stfc.github.io/aiida-mlip/`](https://stfc.github.io/aiida-mlip/)

---

## How to Update Skills

Run the updater at any time:

```bash
# Check upstream for updates without modifying files:
./aiida-mlip/update.sh --check

# Apply updates from GitHub and online docs:
./aiida-mlip/update.sh

# Update from a local checkout:
./aiida-mlip/update.sh --repo /path/to/local/aiida-mlip
```

The updater automatically:
1. Verifies online documentation reachability and version.
2. Clones or pulls the latest repository commit.
3. Introspects AiiDA calculation entry points, data types, and parsers in `pyproject.toml`.
4. Parses calculation classes to extract input and output ports.
5. Regenerates reference documents (`entry_points.md`, `calculations_overview.md`, `ports.md`).
6. Synchronizes example submission scripts.
7. Updates `VERSION` and `plugin.json`.

