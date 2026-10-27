---
name: aiida-mlip-train
description: >-
  Train and fine-tune MLIP models (e.g. MACE) with full provenance tracking using the aiida-mlip plugin (mlip.train).
  Use this skill when preparing datasets, configuring training hyperparameters, or fine-tuning foundation models in AiiDA.
---

# AiiDA-MLIP: Model Training & Fine-Tuning

Train new machine-learned interatomic potentials or fine-tune foundation potentials (such as MACE) with complete hyperparameter, dataset, and checkpoint provenance using AiiDA's `mlip.train` calculation job.

## Key Procedures

### 1. Training a Model from Scratch

```python
from aiida.engine import submit
from aiida.orm import Dict, InstalledCode
from aiida.plugins import CalculationFactory
from aiida_mlip.data.config import JanusConfigfile

code = InstalledCode.collection.get(label="janus")
config_node = JanusConfigfile("train_config.yml")

inputs = {
    "code": code,
    "mlip_config": config_node,
    "metadata": {
        "options": {
            "output_filename": "aiida-stdout.txt",
            "resources": {"num_machines": 1},
        }
    },
}

TrainCalculation = CalculationFactory("mlip.train")
calc_node = submit(TrainCalculation, **inputs)
print(f"Submitted MLIP training: PK={calc_node.pk}")
```

Example `train_config.yml`:
```yaml
name: "mace_model"
train_file: "data/train.xyz"
valid_file: "data/valid.xyz"
test_file: "data/test.xyz"
loss: "universal"
energy_weight: 1.0
forces_weight: 10.0
batch_size: 16
max_num_epochs: 50
device: "cuda"
```

### 2. Fine-Tuning a Foundation Potential

To fine-tune a foundation model (e.g. `mace_mp`), pass `fine_tune=Bool(True)` and attach the pretrained `ModelData` node:
```python
from aiida.orm import Bool
from aiida_mlip.data.model import ModelData

foundation = ModelData.from_local(
    "models/mace_mp_small.model",
    architecture="mace_mp",
)

inputs["fine_tune"] = Bool(True)
inputs["foundation_model"] = foundation

TrainCalculation = CalculationFactory("mlip.train")
calc_node = submit(TrainCalculation, **inputs)
```

### 3. Submitting via CLI Script

```bash
micromamba run -n janus verdi run aiida-mlip/skills/aiida-mlip-train/examples/submit_train.py
```

### 4. Retrieving Trained Model Checkpoint

Once the training calcjob finishes:
```python
from aiida.orm import load_node

calc = load_node(<PK>)
model_node = calc.outputs.model
print(f"Trained model registered: PK={model_node.pk}")
```

See [ports.md](https://github.com/alinelena/ai-rules/blob/main/aiida-mlip/skills/aiida-mlip-train/references/ports.md) for full input/output port definitions.

