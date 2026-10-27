---
name: aiida-mlip-phonons
description: >-
  Calculate harmonic phonon band structures, vibrational density of states (DOS), and thermal properties
  using the aiida-mlip plugin (mlip.ph) and Phonopy.
  Use this skill when analyzing lattice vibrations, dynamical stability, and thermodynamics in AiiDA.
---

# AiiDA-MLIP: Phonon Calculations

Compute harmonic phonon dispersion curves, vibrational density of states (DOS), and thermal properties using Phonopy backed by MLIP force calculations through AiiDA's `mlip.ph` calculation job.

## Key Procedures

### 1. Submitting Phonon Calculation via Python API

```python
from aiida.engine import submit
from aiida.orm import Bool, Float, InstalledCode, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.io import read

atoms = read("structure.cif")
structure_node = StructureData(ase=atoms)
code = InstalledCode.collection.get(label="janus")

inputs = {
    "code": code,
    "architecture": Str("mace_mp"),
    "struct": structure_node,
    "device": Str("cpu"),
    "supercell": Str("2 2 2"),
    "displacement": Float(0.01),  # Å
    "dos": Bool(True),
    "bands": Bool(True),
    "metadata": {
        "options": {
            "resources": {"num_machines": 1},
        }
    },
}

PhononCalculation = CalculationFactory("mlip.ph")
calc_node = submit(PhononCalculation, **inputs)
print(f"Submitted Phonon calculation: PK={calc_node.pk}")
```

### 2. Submitting via CLI Script

```bash
micromamba run -n janus verdi run aiida-mlip/skills/aiida-mlip-phonons/examples/submit_phonon.py \
  "janus@localhost" \
  --struct "structure.cif" \
  --architecture "mace_mp" \
  --supercell "2 2 2" \
  --dos \
  --bands
```

### 3. Extracting Phonon Outputs

```python
from aiida.orm import load_node

calc = load_node(<PK>)
params = calc.outputs.output_parameters.get_dict()
print("Phonon calculation status:", params.get("status"))
```

See [ports.md](https://github.com/alinelena/ai-rules/blob/main/aiida-mlip/skills/aiida-mlip-phonons/references/ports.md) for full input/output port definitions.

