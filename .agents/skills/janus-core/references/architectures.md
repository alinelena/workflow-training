# Supported MLIP Architectures in Janus-Core

| Architecture (`--arch`) | Description | Typical Use Cases |
| :--- | :--- | :--- |
| `mace` | MACE foundation potential | Molecules and materials |
| `mace_mp` | MACE Materials Project potential | Crystalline and inorganic solids |
| `mace_off` | MACE Open Force Field potential | Organic molecules and solution chemistry |
| `chgnet` | Crystal Hamiltonian Graph Neural Network | Minerals, oxides, battery materials |
| `sevennet` | SevenNet graph neural network potential | Inorganic materials, high accuracy |
| `nequip` | E(3)-equivariant interatomic potential | Niche high-precision systems |
| `deepmd` | Deep Potential architecture (DeepMD-kit) | Condensed matter at scale |
| `orb` | Orbital Materials foundation model | Universal materials potential |
| `mattersim` | Microsoft MatterSim foundation potential | Universal materials simulations |
| `grace` | GRACE multi-element potential | Alloys, intermetallics, crystals |
| `upet` | Universal Pretrained Equivariant Transformer | Broad chemical space |
| `fairchem` | FAIRChem / Open Catalyst models | Surface reactions, catalysis, adsorptions |
| `mace_omol` | MACE Open Molecule foundation model | Molecular conformations and reactions |
| `mace_polar` | MACE Polarizable potential | Dielectric and polarization observables |
| `dpa3` | Deep Potential Architecture v3 (alias of deepmd) | DeepMD legacy alias |

## Supported Compute Devices (`--device`)

- `cpu`: Default fallback; runs on host CPU threads.
- `cuda`: NVIDIA GPU acceleration via CUDA.
- `mps`: Apple Silicon GPU acceleration via Metal Performance Shaders.
- `xpu`: Intel GPU acceleration.

## Calculator Construction (`--calc-kwargs`)

Pass calculator-specific keyword arguments as a JSON/Python dictionary string or YAML mapping:
```bash
janus singlepoint --arch mace_mp --calc-kwargs "{'dispersion': True}" --struct NaCl.cif
```
