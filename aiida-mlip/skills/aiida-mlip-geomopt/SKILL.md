---
name: aiida-mlip-geomopt
description: >-
  Submit and manage atomic geometry optimization and unit cell relaxation using the aiida-mlip plugin (mlip.opt).
  Supports cell filters like FrechetCellFilter and trajectory recording.
  Use this skill when minimizing energy or relaxing crystal structures with AiiDA provenance.
---

# AiiDA-MLIP: Geometry Optimization

Optimize atomic coordinates and relax periodic unit cells using MLIPs managed by AiiDA's `mlip.opt` calculation job.

## Key Procedures

### 1. Submitting Coordinate Relaxation (Fixed Cell)

```python
from aiida.engine import submit
from aiida.orm import Bool, Float, InstalledCode, Int, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.io import read

atoms = read("initial.cif")
structure_node = StructureData(ase=atoms)
code = InstalledCode.collection.get(label="janus")

inputs = {
    "code": code,
    "architecture": Str("mace_mp"),
    "structure": structure_node,
    "max_force": Float(0.01),  # eV/Å convergence tolerance
    "steps": Int(500),
    "metadata": {
        "options": {
            "resources": {"num_machines": 1},
        }
    },
}

GeomOptCalculation = CalculationFactory("mlip.opt")
calc_node = submit(GeomOptCalculation, **inputs)
print(f"Submitted GeomOpt: PK={calc_node.pk}")
```

### 2. Full Cell and Coordinate Relaxation

To optimize both coordinates and periodic lattice parameters:
```python
inputs["opt_cell_fully"] = Bool(True)
inputs["filter"] = Str("FrechetCellFilter")

GeomOptCalculation = CalculationFactory("mlip.opt")
calc_node = submit(GeomOptCalculation, **inputs)
```

> [!IMPORTANT]
> Always prefer `"filter": Str("FrechetCellFilter")` for stable metric convergence on periodic unit cells.

### 3. Submitting via CLI Script

```bash
micromamba run -n janus verdi run aiida-mlip/skills/aiida-mlip-geomopt/examples/submit_geomopt.py \
  "janus@localhost" \
  --structure "initial.cif" \
  --architecture "mace_mp" \
  --steps 500 \
  --opt-cell-fully True \
  --device "cpu"
```

### 4. Retrieving Relaxed Structure

```python
from aiida.orm import load_node

calc = load_node(<PK>)
relaxed_struct = calc.outputs.final_structure
print(f"Relaxed Cell: {relaxed_struct.cell}")
print(f"Final Energy: {calc.outputs.output_parameters['final_energy']} eV")
```

See [ports.md](https://github.com/alinelena/ai-rules/blob/main/aiida-mlip/skills/aiida-mlip-geomopt/references/ports.md) for full input/output port definitions.

