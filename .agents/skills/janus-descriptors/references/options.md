# Janus Descriptors Options Reference

The `janus descriptors` command computes invariant machine learned representations from interatomic potentials.

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--arch` | `string` | *required* | MLIP architecture (`mace_mp`, `mace`, etc.) |
| `--struct` | `path` | *required* | Path to structure |
| `--invariants-only / --no-invariants-only` | `bool` | `True` | Only calculate invariant descriptors |
| `--calc-per-atom` | `bool` | `False` | Calculate descriptors for each individual atom |
| `--calc-per-element` | `bool` | `False` | Calculate mean descriptors averaged per chemical element |
| `--out` | `path` | auto | Path to save output structure with computed descriptors |
| `--device` | `string` | `cpu` | Device (`cpu`, `cuda`, `mps`, `xpu`) |
| `--model` | `path / str` | `None` | Path to custom checkpoint file or model identifier |
| `--file-prefix` | `path` | `./janus_results/<name>` | Output prefix and directory |
| `--tracker / --no-tracker`| `bool` | `True` | Enable/disable CodeCarbon tracking |
| `--log` | `path` | auto | Log file path |
| `--summary` | `path` | auto | Summary YAML file path |

