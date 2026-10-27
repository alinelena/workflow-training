---
name: aiida-mlip-md
description: >-
  Run molecular dynamics (MD) simulations across NVE, NVT, and NPT ensembles using the aiida-mlip plugin (mlip.md).
  Tracks trajectories, temperature, and pressure with full AiiDA provenance.
  Use this skill when submitting MD simulations or sampling phase space with MLIPs in AiiDA.
---

# AiiDA-MLIP: Molecular Dynamics (MD)

Simulate finite-temperature dynamics of atomic systems across microcanonical, canonical, and isothermal-isobaric ensembles with automated trajectory tracking in AiiDA's `mlip.md` calculation job.

## Supported Ensembles

- `nve`: Constant energy and volume.
- `nvt`: Canonical ensemble (Langevin or Nosé-Hoover).
- `npt`: Isothermal-isobaric ensemble (variable volume).

---

## Key Procedures

### 1. Submitting NVT Langevin Dynamics

```python
from aiida.engine import submit
from aiida.orm import Dict, InstalledCode, Str, StructureData
from aiida.plugins import CalculationFactory
from ase.io import read

atoms = read("structure.cif")
structure_node = StructureData(ase=atoms)
code = InstalledCode.collection.get(label="janus")

md_parameters = {
    "temp": 300.0,        # K
    "timestep": 1.0,      # fs
    "steps": 10000,
    "thermostat": "langevin",
    "friction": 0.01,
    "stats-every": 10,
    "traj-every": 50,
}

inputs = {
    "code": code,
    "architecture": Str("mace_mp"),
    "structure": structure_node,
    "ensemble": Str("nvt"),
    "md_dict": Dict(md_parameters),
    "metadata": {
        "options": {
            "resources": {"num_machines": 1},
        }
    },
}

MDCalculation = CalculationFactory("mlip.md")
calc_node = submit(MDCalculation, **inputs)
print(f"Submitted MD calculation: PK={calc_node.pk}")
```

### 2. Submitting with Config Node (`submit_md_using_config.py`)

```python
from aiida.engine import run_get_node
from aiida.orm import InstalledCode
from aiida.plugins import CalculationFactory
from aiida_mlip.data.config import JanusConfigfile

code = InstalledCode.collection.get(label="janus")
config_node = JanusConfigfile("nvt_config.yml")

MDCalculation = CalculationFactory("mlip.md")
results, node = run_get_node(
    MDCalculation,
    code=code,
    config=config_node,
    metadata={"options": {"resources": {"num_machines": 1}}},
)
```

### 3. Submitting via CLI Script

```bash
micromamba run -n janus verdi run aiida-mlip/skills/aiida-mlip-md/examples/submit_md.py \
  "janus@localhost" \
  --structure "structure.cif" \
  --architecture "mace_mp" \
  --ensemble "nvt" \
  --md-dict-str "{'temp': 300, 'steps': 1000, 'traj-every': 50, 'stats-every': 10}"
```

See [ports.md](https://github.com/alinelena/ai-rules/blob/main/aiida-mlip/skills/aiida-mlip-md/references/ports.md) for full input/output port definitions.

