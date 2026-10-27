# Janus Nudged Elastic Band (NEB) Options Reference

The `janus neb` command calculates minimum energy pathways (MEP) and activation barriers connecting two known states.

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--arch` | `string` | *required* | MLIP architecture (`mace_mp`, `sevennet`, etc.) |
| `--init-struct` | `path` | `None` | Path to initial endpoint structure |
| `--final-struct` | `path` | `None` | Path to final endpoint structure |
| `--neb-structs` | `path` | `None` | Path to file containing all pre-interpolated NEB images |
| `--n-images` | `int` | `5` | Number of intermediate images between endpoints |
| `--neb-class` | `string` | `NEB` | ASE NEB class (`NEB`, `DyNEB`) |
| `--climb / --no-climb` | `bool` | `False` | Enable Climbing Image NEB (CI-NEB) for saddle point refinement |
| `--interpolator` | `string` | `idpp` | Image interpolation algorithm (`idpp`, `linear`, `ase`, `pymatgen`) |
| `--fmax` | `float` | `0.05` | Convergence force tolerance on images (eV/Å) |
| `--steps` | `int` | `500` | Maximum optimization steps |
| `--optimizer` | `string` | `FIRE` | Optimizer algorithm (`FIRE`, `LBFGS`, `BFGS`) |
| `--device` | `string` | `cpu` | Device (`cpu`, `cuda`, `mps`, `xpu`) |
| `--file-prefix` | `path` | `./janus_results/<name>` | Output prefix and directory |
| `--tracker / --no-tracker`| `bool` | `True` | Enable/disable CodeCarbon tracking |
| `--log` | `path` | auto | Log file path |
| `--summary` | `path` | auto | Summary YAML file path |

