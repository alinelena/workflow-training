---
name: aiida-mlip-singlepoint
description: >-
  Submit and manage static single-point calculations (energy, forces, stress, Hessians)
  using the aiida-mlip plugin (mlip.sp).
  Use this skill when evaluating potential energy, atomic forces, or stress tensors with AiiDA provenance.
---

# AiiDA-MLIP: Single Point Calculations

Run static single-point energy, forces, stress, and Hessian evaluations using machine-learned interatomic potentials through AiiDA's `mlip.sp` calculation job.

## Key Procedures

### 1. Submitting via Python API

```python
from aiida.engine import submit
from aiida.orm import Dict, InstalledCode, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.io import read

# Load structure and code
atoms = read("structure.cif")
structure_node = StructureData(ase=atoms)
code = InstalledCode.collection.get(label="janus")

inputs = {
    "code": code,
    "architecture": Str("mace_mp"),
    "structure": structure_node,
    "device": Str("cpu"),
    "metadata": {
        "options": {
            "resources": {"num_machines": 1},
        }
    },
}

SinglePointCalculation = CalculationFactory("mlip.sp")
calc_node = submit(SinglePointCalculation, **inputs)
print(f"Submitted singlepoint calculation: PK={calc_node.pk}")
```

### 2. Submitting with a YAML Config Node (`JanusConfigfile`)

Group simulation parameters in a tracked YAML config node:
```python
from aiida.engine import run_get_node
from aiida.orm import InstalledCode
from aiida.plugins import CalculationFactory
from aiida_mlip.data.config import JanusConfigfile

code = InstalledCode.collection.get(label="janus")
config_node = JanusConfigfile("singlepoint_config.yml")

SinglePointCalculation = CalculationFactory("mlip.sp")
results, node = run_get_node(
    SinglePointCalculation,
    code=code,
    config=config_node,
    metadata={"options": {"resources": {"num_machines": 1}}},
)

print(f"Calculated Energy: {results['output_parameters']['energy']} eV")
```

### 3. Submitting via CLI Script

Execute the provided submission script with `verdi run`:
```bash
micromamba run -n janus verdi run aiida-mlip/skills/aiida-mlip-singlepoint/examples/submit_singlepoint.py \
  "janus@localhost" \
  --structure "structure.cif" \
  --architecture "mace_mp" \
  --device "cpu"
```

### 4. Inspecting Results

```bash
micromamba run -n janus verdi process list -a
micromamba run -n janus verdi process show <PK>
micromamba run -n janus verdi calcjob outputcat <PK>
```

See [ports.md](https://github.com/alinelena/ai-rules/blob/main/aiida-mlip/skills/aiida-mlip-singlepoint/references/ports.md) for full input/output port definitions.

