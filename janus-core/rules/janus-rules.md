# Janus-Core Multi-Agent Execution Rules

When interacting with STFC `janus-core` or running atomistic calculations:

1. **Python Environment**:
   - Always run commands in the `janus` micromamba environment (`micromamba run -n janus python`) or via `uv run --no-project python`.
   - Install packages and run tests using `uv`.

2. **Headless Execution & Emission Tracking**:
   - In automated scripts or test pipelines, pass `--no-tracker` to prevent unnecessary CodeCarbon overhead or background pynvml warnings.

3. **Cell Relaxation Filter**:
   - For crystal geometry optimizations involving unit cell changes, always prefer `--filter FrechetCellFilter` (or `filter_func="FrechetCellFilter"`) over older unit cell filters for stable metric convergence.

4. **Structure Verification**:
   - Always check structure coordinates and periodic boundary conditions using `janus-core/tools/structure_inspector.py <structure>` before executing heavy simulations.

5. **Live Documentation**:
   - For up-to-date documentation and snippets, use Context7 via `janus_mcp.context7_client.Context7Client` or query `https://context7.com/stfc/janus-core` and `https://stfc.github.io/janus-core/`.

