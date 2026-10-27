# Janus MLIP Preprocessing Options Reference

The `janus preprocess` command converts atomic datasets into HDF5 binary representations for training efficiency.

## Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mlip-config` | `path` | *required* | Path to configuration file passed to MLIP CLI |
| `--log` | `path` | `./janus_results/preprocess-log.yml` | Path to save preprocessing logs |
| `--tracker / --no-tracker`| `bool` | `True` | Enable/disable CodeCarbon tracking |
| `--summary` | `path` | `./janus_results/preprocess-summary.yml` | Path to save preprocessing summary |

