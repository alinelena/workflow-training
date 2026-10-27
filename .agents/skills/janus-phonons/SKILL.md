---
name: janus-phonons
description: >-
  Calculate phonon band structures, vibrational density of states (DOS), and thermal properties
  (heat capacity, vibrational entropy, zero-point energy) using STFC janus-core and Phonopy.
  Use this skill when analyzing lattice dynamics, phonon stability, or vibrational thermodynamics.
---

# Janus-Core: Phonon Calculations

Compute harmonic phonon dispersion curves, vibrational density of states (DOS), projected DOS (PDOS), and thermal properties of periodic crystalline solids using Phonopy backed by MLIP force calculations.

## Key Procedures

### 1. Basic Phonon Calculation (DOS & Band Structure)

Generates finite displacements on a supercell, evaluates forces with MLIP, and calculates phonon band structure and DOS:
```bash
janus phonons \
  --arch mace_mp \
  --struct structure.cif \
  --supercell "2 2 2" \
  --displacement 0.01 \
  --mesh "20 20 20" \
  --device cpu \
  --no-tracker
```

### 2. Computing Temperature-Dependent Thermal Properties

Evaluates vibrational Helmholtz free energy ($F$), entropy ($S$), and constant-volume heat capacity ($C_v$) across a temperature range:
```bash
janus phonons \
  --arch mace_mp \
  --struct structure.cif \
  --supercell "2 2 2" \
  --temp-min 0 \
  --temp-max 1000 \
  --temp-step 50 \
  --calcs thermal \
  --calcs dos \
  --device cpu \
  --no-tracker
```

### 3. Pre-Optimization Before Phonon Calculations

Lattice dynamics calculations require zero residual forces on atoms:
```bash
janus phonons \
  --arch mace_mp \
  --struct structure.cif \
  --minimize \
  --supercell "2 2 2" \
  --symmetrize \
  --no-tracker
```

### 4. Python API Example

```python
from ase.io import read
from janus_core.calculations.phonons import Phonons

atoms = read("structure.cif")
phonon_calc = Phonons(
    struct=atoms,
    arch="mace_mp",
    supercell="2 2 2",
    displacement=0.01,
    calcs=["bands", "dos", "thermal"],
    temp_min=0,
    temp_max=800,
    temp_step=20,
)
phonon_calc.calc()
print("Phonon calculation completed.")
```

See [options.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-phonons/references/options.md) for full options reference.

