# Janus Elasticity Options Reference

The `janus elasticity` command applies small strains to determine the 6x6 elasticity stiffness matrix $C_{ij}$ and polycrystalline Voigt-Reuss-Hill moduli.

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--arch` | `string` | *required* | MLIP architecture (`mace_mp`, `sevennet`, etc.) |
| `--struct` | `path` | *required* | Path to structure |
| `--shear-magnitude` | `float` | `0.06` | Magnitude of shear strain applied to cell |
| `--normal-magnitude`| `float` | `0.01` | Magnitude of normal strain applied to cell |
| `--n-strains` | `int` | `4` | Number of normal and shear strains |
| `--write-voigt` | `bool` | `False` | Save structure with Voigt stress results |
| `--minimize` | `bool` | `False` | Minimize initial structure before calculating elasticity |
| `--device` | `string` | `cpu` | Device (`cpu`, `cuda`, `mps`, `xpu`) |
| `--file-prefix` | `path` | `./janus_results/<name>` | Directory and prefix for outputs |
| `--tracker / --no-tracker`| `bool` | `True` | Enable/disable CodeCarbon tracking |
| `--log` | `path` | auto | Log file path |
| `--summary` | `path` | auto | Summary YAML file path |

