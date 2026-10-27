---
name: aiida-mlip-workflows
description: >-
  Build and execute high-throughput (HT) calculation workgraphs using aiida-mlip and aiida-workgraph.
  Automates multi-structure screening, job concurrency limits, and linked simulation pipelines.
  Use this skill when running batch MLIP simulations or complex calculation workflows in AiiDA.
---

# AiiDA-MLIP: Workflows & High-Throughput Workgraphs

Automate large-scale materials screening and multi-stage simulation pipelines using `aiida-mlip` integrated with `aiida-workgraph`.

## Key Procedures

### 1. High-Throughput Batch Calculation over a Directory of Structures

`aiida_mlip.workflows.ht_workgraph.build_ht_calc` scans a directory containing structures and builds a workgraph executing calculations concurrently up to a specified job limit:

```python
from pathlib import Path
from aiida.orm import InstalledCode
from aiida.plugins import CalculationFactory
from aiida_mlip.data.model import ModelData
from aiida_mlip.workflows.ht_workgraph import build_ht_calc

# Select calculation type (e.g. SinglePoint, GeomOpt)
SinglepointCalc = CalculationFactory("mlip.sp")

# Common calculation inputs
inputs = {
    "model": ModelData.from_local(
        "models/mace_mp_small.model",
        architecture="mace_mp",
    ),
    "metadata": {
        "options": {"resources": {"num_machines": 1}},
    },
    "code": InstalledCode.collection.get(label="janus"),
}

# Construct WorkGraph across all structures in a folder
wg = build_ht_calc(
    calc=SinglepointCalc,
    folder=Path("./structures/"),
    calc_inputs=inputs,
    final_struct_key="xyz_output",
)

# Set maximum concurrent active jobs
wg.max_number_jobs = 10

# Submit the WorkGraph to the AiiDA engine
wg.submit()
print("High-throughput workgraph submitted successfully.")
```

### 2. Multi-Stage Pipelines (Relaxation -> Phonons)

Chain calculation outputs into downstream stages within an AiiDA WorkGraph:
- Stage 1: `mlip.opt` relaxes the candidate structure.
- Stage 2: `mlip.ph` takes the relaxed structure node and computes phonon band dispersion and DOS.
- Automatic failure handling and provenance logging for every branch.

### 3. Monitoring WorkGraph Progress

```bash
micromamba run -n janus verdi process list -a
micromamba run -n janus verdi process status <WorkGraph-PK>
```

See the submission script example in [submit_ht_workgraph.py](https://github.com/alinelena/ai-rules/blob/main/aiida-mlip/skills/aiida-mlip-workflows/examples/submit_ht_workgraph.py).

