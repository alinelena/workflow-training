# Janus SinglePoint Options Reference

The `janus singlepoint` command performs static calculations to compute total potential energy, forces, stress, and Hessians.

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--arch` | `string` | *required* | MLIP architecture (`mace_mp`, `sevennet`, `chgnet`, `nequip`, etc.) |
| `--struct` | `path` | *required* | Path to structure file (CIF, XYZ, POSCAR, etc.) |
| `--properties` | `list[str]` | `[energy, forces, stress]` | Properties to calculate: `energy`, `forces`, `stress`, `hessian` |
| `--out` | `path` | auto | Path to save evaluated structures (e.g. `.extxyz`) |
| `--device` | `string` | `cpu` | Device (`cpu`, `cuda`, `mps`, `xpu`) |
| `--model` | `path / str` | `None` | Path to custom checkpoint file or HuggingFace ID |
| `--calc-kwargs` | `dict` | `{}` | Arguments passed to underlying calculator constructor |
| `--read-kwargs` | `dict` | `{'index': ':'}` | Arguments passed to `ase.io.read` |
| `--write-kwargs` | `dict` | `{}` | Arguments passed to `ase.io.write` |
| `--file-prefix` | `path` | `./janus_results/<name>` | Directory and prefix for outputs |
| `--tracker / --no-tracker` | `bool` | `True` | Enable/disable CodeCarbon emissions tracking |
| `--log` | `path` | auto | Filepath for execution log |
| `--summary` | `path` | auto | Filepath for execution summary YAML |

