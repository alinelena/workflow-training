# Janus Molecular Dynamics (MD) Options Reference

The `janus md` command simulates finite-temperature dynamics across statistical mechanical ensembles.

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--arch` | `string` | *required* | MLIP architecture (`mace_mp`, `sevennet`, `chgnet`, etc.) |
| `--struct` | `path` | *required* | Initial atomic structure |
| `--ensemble` | `string` | *required* | Thermodynamic ensemble: `nvt`, `npt`, `nve`, `nvt-nh`, `nvt-csvr`, `npt-mtk`, `npt-mtk-iso`, `npt-mtk-aniso`, `nph` |
| `--steps` | `int` | `0` | Number of MD integration steps |
| `--timestep` | `float` | `1.0` | Integrator timestep in femtoseconds (fs) |
| `--temp` | `float` | `300.0` | Target temperature in Kelvin (K) |
| `--thermostat-time` | `float` | `50.0` | Thermostat coupling time in fs |
| `--barostat-time` | `float` | `75.0` | Barostat coupling time in fs (NPT) |
| `--pressure` | `float` | `1.01325 bar` | Target pressure (in bar or GPa) |
| `--friction` | `float` | `0.005` | Langevin thermostat friction coefficient (fs⁻¹) |
| `--equil-steps` | `int` | `0` | Equilibration steps before production run |
| `--rescale-velocities` | `bool` | `False` | Rescale velocities during equilibration |
| `--remove-rot` | `bool` | `False` | Remove net rotation (molecules/clusters) |
| `--stats-every` | `int` | `1` | Frequency to record thermodynamic statistics |
| `--traj-every` | `int` | `100` | Frequency to save trajectory snapshots |
| `--traj-start` | `int` | `0` | First step to start writing trajectory |
| `--restart-every` | `int` | `1000` | Frequency to write restart snapshots |
| `--temp-start` | `float` | `None` | Starting temperature for temperature ramp (K) |
| `--temp-end` | `float` | `None` | Ending temperature for temperature ramp (K) |
| `--temp-step` | `float` | `None` | Step change in temperature during ramp |
| `--temp-time` | `float` | `None` | Time per temperature ramp increment (fs) |
| `--plumed-input` | `path` | `None` | Path to PLUMED input file for metadynamics |
| `--device` | `string` | `cpu` | Device (`cpu`, `cuda`, `mps`, `xpu`) |
| `--file-prefix` | `path` | `./janus_results/<name>` | Directory and prefix for outputs |
| `--tracker / --no-tracker`| `bool` | `True` | Enable/disable CodeCarbon tracking |
| `--log` | `path` | auto | Execution log filepath |
| `--summary` | `path` | auto | Execution summary YAML filepath |

