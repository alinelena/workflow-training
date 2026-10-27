# Janus Equation of State (EOS) Options Reference

The `janus eos` command calculates energy-volume curves to determine equilibrium volume, equilibrium energy, and bulk modulus.

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--arch` | `string` | *required* | MLIP architecture (`mace_mp`, `sevennet`, etc.) |
| `--struct` | `path` | *required* | Path to structure |
| `--min-volume` | `float` | `0.95` | Minimum volume scale factor |
| `--max-volume` | `float` | `1.05` | Maximum volume scale factor |
| `--n-volumes` | `int` | `7` | Number of volume points to sample |
| `--eos-type` | `string` | `birchmurnaghan` | EOS equation fit: `birchmurnaghan`, `murnaghan`, `vinet`, `pouriertarantola`, `antonschmidt`, `taylor` |
| `--minimize` | `bool` | `False` | Minimize initial structure before EOS calculation |
| `--device` | `string` | `cpu` | Device (`cpu`, `cuda`, `mps`, `xpu`) |
| `--file-prefix` | `path` | `./janus_results/<name>` | Directory and prefix for outputs |
| `--tracker / --no-tracker`| `bool` | `True` | Enable/disable CodeCarbon tracking |
| `--log` | `path` | auto | Log file path |
| `--summary` | `path` | auto | Summary YAML file path |

