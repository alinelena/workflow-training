# Janus GeomOpt Options Reference

The `janus geomopt` command performs atomic coordinate and unit cell optimization to reach local potential energy minima.

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--arch` | `string` | *required* | MLIP architecture (`mace_mp`, `sevennet`, `chgnet`, etc.) |
| `--struct` | `path` | *required* | Path to structure to optimize |
| `--optimizer` | `string` | `LBFGS` | Name of ASE optimizer function (`LBFGS`, `BFGS`, `FIRE`) |
| `--fmax` | `float` | `0.1` | Maximum force tolerance for convergence (eV/Å) |
| `--steps` | `int` | `1000` | Maximum number of optimization steps |
| `--opt-cell-lengths` | `bool` | `False` | Optimize cell lengths/vectors with hydrostatic strain |
| `--opt-cell-fully` | `bool` | `False` | Fully optimize cell vectors, angles, and atomic positions |
| `--filter` | `string` | `FrechetCellFilter` | ASE cell filter (used with `--opt-cell-lengths` or `--opt-cell-fully`) |
| `--pressure` | `float` | `0.0` | Scalar target pressure in GPa |
| `--fix-symmetry` | `bool` | `False` | Fix symmetry during optimization |
| `--symmetrize` | `bool` | `False` | Symmetrize structure after optimization |
| `--symmetry-tolerance`| `float` | `1e-5` | Symmetry tolerance threshold |
| `--out` | `path` | auto | Path to save optimized structure |
| `--write-traj` | `bool` | `False` | Whether to save optimization trajectory |
| `--traj` | `path` | auto | Filepath for optimization trajectory |
| `--device` | `string` | `cpu` | Device (`cpu`, `cuda`, `mps`, `xpu`) |
| `--file-prefix` | `path` | `./janus_results/<name>` | Output prefix and directory |
| `--tracker / --no-tracker`| `bool` | `True` | Enable/disable CodeCarbon tracking |
| `--log` | `path` | auto | Log file path |
| `--summary` | `path` | auto | Summary YAML file path |

