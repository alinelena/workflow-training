# Janus MLIP Training Options Reference

The `janus train` command trains or fine-tunes MLIP models by passing structured configurations to the underlying MLIP engine (e.g. MACE).

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--arch` | `string` | *required* | MLIP architecture to train (`mace`, `mace_mp`, etc.) |
| `--mlip-config` | `path` | *required* | Path to MLIP training configuration file (YAML) |
| `--fine-tune / --no-fine-tune` | `bool` | `False` | Whether to fine-tune an existing foundational model |
| `--file-prefix` | `path` | `./janus_results` | Directory and prefix for checkpoints and logs |
| `--tracker / --no-tracker`| `bool` | `True` | Enable/disable CodeCarbon emissions tracking |
| `--log` | `path` | auto | Execution log filepath |
| `--summary` | `path` | auto | Execution summary YAML filepath |

