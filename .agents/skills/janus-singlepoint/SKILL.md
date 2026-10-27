---
name: janus-singlepoint
description: >-
  Calculate static single-point energies, forces, stress tensors, Hessians, and MLIP descriptors
  for molecular and periodic crystal structures using STFC janus-core.
  Use this skill when evaluating potential energies, atomic forces, or descriptors of an atomic structure.
---

# Janus-Core: Single Point Calculations

Perform static single-point calculations on one or more atomic structures to compute total potential energy, forces on individual atoms, Cauchy stress tensor, and Hessian matrices using machine learned interatomic potentials.

## Key Procedures

### 1. Basic CLI Command

Evaluate energy, forces, and stress on a crystal or molecular structure:
```bash
janus singlepoint \
  --arch mace_mp \
  --struct structure.cif \
  --device cpu \
  --no-tracker
```

### 2. Requesting Specific Properties

Request explicit physical observables and save output to an extended XYZ file:
```bash
janus singlepoint \
  --arch mace_mp \
  --struct structure.xyz \
  --properties energy \
  --properties forces \
  --properties stress \
  --properties hessian \
  --out results.extxyz \
  --no-tracker
```

### 3. Using Python API

```python
from ase.io import read
from janus_core.calculations.single_point import SinglePoint

# Load structure
atoms = read("structure.cif")

# Initialize and run single-point calculation
sp = SinglePoint(
    struct=atoms,
    arch="mace_mp",
    device="cpu",
    properties=["energy", "forces", "stress"],
)
results = sp.calc()

print(f"Potential Energy: {results['energy']} eV")
print(f"Max Force: {abs(results['forces']).max()} eV/Å")
```

### 4. Configuration File Workflow

Run with a declarative YAML configuration file:
```bash
janus singlepoint --config singlepoint_config.yml
```

Example `singlepoint_config.yml`:
```yaml
arch: mace_mp
device: cpu
struct: structure.cif
properties:
  - energy
  - forces
  - stress
out: results.extxyz
tracker: false
```

See [options.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-singlepoint/references/options.md) for the complete list of options.

