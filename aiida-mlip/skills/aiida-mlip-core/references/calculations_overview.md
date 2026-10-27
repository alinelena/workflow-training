# AiiDA-MLIP Calculations Overview

`aiida-mlip` executes `janus-core` calculations within the AiiDA automated provenance workflow system.

## Supported Calculation Types

| Calculation | Entry Point | Purpose |
| :--- | :--- | :--- |
| Single Point | `mlip.sp` | Static energy, forces, stress, Hessians |
| Geometry Optimization | `mlip.opt` | Force minimization & cell relaxation |
| Molecular Dynamics | `mlip.md` | NVE, NVT, NPT finite temperature dynamics |
| Phonons | `mlip.ph` | Phonon band structures, DOS, and thermodynamics |
| Descriptors | `mlip.descriptors` | MLIP invariant representations extraction |
| Train / Fine-Tune | `mlip.train` | MACE and MLIP model training and transfer learning |

## Common Input Ports Across All Calculations

- `code` (`InstalledCode`): The registered `janus` executable (e.g. `janus@localhost`).
- `metadata.options.resources`: Execution machine resources (e.g. `{'num_machines': 1}`).
- `structure` (`StructureData` or `XyzData`): The target atomic configuration.
- `architecture` (`Str`): MLIP architecture (`mace_mp`, `sevennet`, `chgnet`, etc.).
- `model` (`ModelData`): Checkpoint file or URI to trained model weights.
- `config` (`JanusConfigfile`): Optional YAML configuration node supplying calculation arguments.
