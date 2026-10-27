# `janus-core` — AI Skills & Rules Group for Antigravity

This directory contains a complete, self-contained capability group of **AI Skills**, **Execution Rules**, **Reference Documentation**, and **Tools** for STFC [`janus-core`](https://github.com/stfc/janus-core) (version 0.9.8+).

---

## What Is Included

### 1. Skills Group (`janus-core/skills/`)

| Skill | Slash Command | Capabilities & Scope |
| :--- | :--- | :--- |
| **`janus-core`** | `/janus-core` | Foundation guide: MLIP architectures (`mace_mp`, `sevennet`, `chgnet`, etc.), devices (`cpu`, `cuda`, `mps`, `xpu`), YAML configuration workflows, and Context7 live documentation. |
| **`janus-singlepoint`** | `/janus-singlepoint` | Static single-point energy, forces, Cauchy stress tensors, and Hessians. |
| **`janus-geomopt`** | `/janus-geomopt` | Geometry optimization and cell relaxation (`FrechetCellFilter`, hydrostatic strain, external pressure). |
| **`janus-md`** | `/janus-md` | Molecular dynamics simulations across NVE, NVT, NPT ensembles, temperature ramps, and PLUMED metadynamics. |
| **`janus-eos-elasticity`**| `/janus-eos-elasticity`| Equation of state ($E-V$ fits, bulk modulus) and full 6x6 elasticity stiffness matrix ($C_{ij}$, VRH moduli). |
| **`janus-phonons`** | `/janus-phonons` | Phonon band structures, DOS, PDOS, and vibrational thermodynamics via Phonopy integration. |
| **`janus-neb`** | `/janus-neb` | Nudged Elastic Band calculation with Climbing Image (CI-NEB), IDPP image interpolation, and activation barriers. |
| **`janus-descriptors`** | `/janus-descriptors` | MLIP invariant representations and descriptors (per-structure, per-element, per-atom). |
| **`janus-train-mlip`** | `/janus-train-mlip` | MLIP model training, fine-tuning, and binary HDF5 dataset preprocessing. |
| **`janus-analysis`** | `/janus-analysis` | MD trajectory observables (RDF $g(r)$, VAF/VACF, MSD, diffusion coefficients) and correlators. |

### 2. Multi-Agent Execution Rules (`janus-core/rules/`)

- `janus-rules.md` / `AGENTS.md`: Enforces key execution guidelines:
  - Run in the `janus` micromamba environment (`micromamba run -n janus python`) or via `uv`.
  - Pass `--no-tracker` in headless / CI environments to avoid unnecessary emissions logging overhead.
  - Always use `--filter FrechetCellFilter` for periodic cell relaxations.
  - Inspect atomic configurations with `janus-core/tools/structure_inspector.py` prior to heavy simulations.
  - Live documentation references (`https://stfc.github.io/janus-core/`).

### 3. Tools (`janus-core/tools/`)

- `structure_inspector.py`: Fast CLI and JSON inspector for atomic structures (CIF, XYZ, POSCAR, PDB), reporting chemical formula, atom count, periodicity, lattice parameters, and cell volume.

### 4. Update Mechanism (`janus-core/update.sh` and `janus-core/scripts/update_skills.py`)

A zero-dependency update mechanism that keeps all skills and reference documentation synchronized with:
- **Upstream GitHub Repository**: [`https://github.com/stfc/janus-core`](https://github.com/stfc/janus-core)
- **Official Documentation**: [`https://stfc.github.io/janus-core/`](https://stfc.github.io/janus-core/)

---

## How to Update Skills

Run the updater at any time:

```bash
# Check upstream for updates without modifying files:
./janus-core/update.sh --check

# Apply updates from GitHub and online docs:
./janus-core/update.sh

# Update from a local checkout:
./janus-core/update.sh --repo /path/to/local/janus-core
```

The updater automatically:
1. Verifies online documentation reachability and version.
2. Clones or pulls the latest repository commit.
3. Introspects Typer CLI definitions, supported MLIP architectures, compute devices, and ensembles.
4. Regenerates reference documents (`cli_overview.md`, `architectures.md`, `options.md`).
5. Synchronizes full YAML configuration examples.
6. Updates `VERSION` and `plugin.json`.

