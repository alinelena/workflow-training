# Janus-Core Rules for Agents

## Execution Guidelines

1. **Python Environment**:
   - Run commands in the `janus` environment (`micromamba run -n janus python`) or via `uv run --no-project python`.
   - Install packages and run tests using `uv`.

2. **Headless Execution & Emission Tracking**:
   - Pass `--no-tracker` in CLI commands and automated scripts to disable CodeCarbon emission tracking and background pynvml warnings.

3. **Cell Relaxation Filter**:
   - For crystal geometry optimizations with variable cell dimensions, use `--filter FrechetCellFilter` (or `filter_func="FrechetCellFilter"`).

4. **Structure Verification**:
   - Always verify cell parameters, pbc flags, and atomic compositions using `janus-core/tools/structure_inspector.py <structure>` prior to running extensive MD or optimization workflows.

5. **Upstream Documentation**:
   - Reference `https://stfc.github.io/janus-core/` and `https://github.com/stfc/janus-core`.

