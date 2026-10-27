---
name: janus-train-mlip
description: >-
  Preprocess atomistic datasets and train or fine-tune machine-learned interatomic potentials (MLIPs)
  such as MACE using STFC janus-core.
  Use this skill when preparing dataset splits, converting data to HDF5, or fine-tuning MLIP models.
---

# Janus-Core: MLIP Training & Fine-Tuning

Preprocess atomistic training datasets and train or fine-tune foundation interatomic potentials (e.g. MACE) with emission tracking and logging.

## Key Procedures

### 1. Preprocess Training Data

Converts raw atomic dataset files (XYZ) into binary HDF5 format to accelerate training epochs:
```bash
janus preprocess \
  --mlip-config preprocess_config.yml \
  --no-tracker
```

Example `preprocess_config.yml`:
```yaml
train_file: "data/train.xyz"
valid_file: "data/val.xyz"
statistics_file: "data/stats.json"
h5_prefix: "data/processed_"
```

### 2. Fine-Tuning a Foundation Potential

Fine-tune a pretrained foundation model (e.g. `mace_mp`) on a specific material or chemical domain:
```bash
janus train \
  --arch mace \
  --mlip-config train_config.yml \
  --fine-tune \
  --file-prefix ./models/finetuned_mace \
  --no-tracker
```

Example `train_config.yml`:
```yaml
name: "mace_finetuned"
train_file: "data/train.xyz"
valid_file: "data/val.xyz"
model: "mace_mp_small.model"
loss: "energy_forces"
energy_weight: 1.0
forces_weight: 10.0
batch_size: 16
max_num_epochs: 50
device: "cuda"
```

### 3. Training from Scratch

```bash
janus train \
  --arch mace \
  --mlip-config train_scratch_config.yml \
  --file-prefix ./models/scratch_mace \
  --no-tracker
```

See [train_options.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-train-mlip/references/train_options.md) and [preprocess_options.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-train-mlip/references/preprocess_options.md) for full options.

