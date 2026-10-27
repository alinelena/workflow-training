# Janus Phonons Options Reference

The `janus phonons` command calculates harmonic phonon dispersions, vibrational density of states (DOS), and thermal properties via Phonopy integration.

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--arch` | `string` | *required* | MLIP architecture (`mace_mp`, `sevennet`, etc.) |
| `--struct` | `path` | *required* | Path to structure |
| `--supercell` | `string` | `"2 2 2"` | Supercell matrix in Phonopy format (`"2"`, `"2 2 2"`, or 9 ints) |
| `--displacement` | `float` | `0.01` | Atomic displacement distance for force constants (Å) |
| `--mesh` | `string` | `"20 20 20"` | Uniform q-point mesh for Brillouin zone integration |
| `--calcs` | `list[str]`| `None` | Calculations to perform: `bands`, `dos`, `pdos`, `thermal`, `qpoints` |
| `--symmetrize` | `bool` | `False` | Symmetrize force constants using crystal symmetry |
| `--temp-min` | `float` | `0.0` | Minimum temperature for thermal calculations (K) |
| `--temp-max` | `float` | `1000.0` | Maximum temperature for thermal calculations (K) |
| `--temp-step` | `float` | `10.0` | Temperature step increment for thermal properties (K) |
| `--minimize` | `bool` | `False` | Minimize structure prior to phonon displacement calculations |
| `--device` | `string` | `cpu` | Device (`cpu`, `cuda`, `mps`, `xpu`) |
| `--file-prefix` | `path` | `./janus_results/<name>` | Directory and prefix for outputs |
| `--tracker / --no-tracker`| `bool` | `True` | Enable/disable CodeCarbon tracking |
| `--log` | `path` | auto | Log file path |
| `--summary` | `path` | auto | Summary YAML file path |

