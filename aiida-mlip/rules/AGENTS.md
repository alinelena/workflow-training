# AiiDA-MLIP Rules for Agents

## Execution Guidelines

1. **Python Environment**:
   - Run commands in the `janus` environment (`micromamba run -n janus python` or `micromamba run -n janus verdi ...`) or via `uv run --no-project python`.
   - Install packages and run tests using `uv`.

2. **AiiDA Services**:
   - Check `verdi daemon status` before submitting workflows.
   - Reference `janus@localhost` as the default execution code.

3. **Data Provenance**:
   - Use `CalculationFactory` with registered entry points (`mlip.sp`, `mlip.opt`, `mlip.md`, `mlip.ph`, `mlip.descriptors`, `mlip.train`).
   - Group configuration parameters in `JanusConfigfile` or AiiDA `Dict` nodes for full provenance tracking.

4. **Emissions Tracking**:
   - Set `tracker: false` in YAML configuration files to prevent unnecessary background tracking.

5. **Upstream Documentation**:
   - Reference [AiiDA-MLIP Documentation](https://stfc.github.io/aiida-mlip/) and [AiiDA-MLIP Repository](https://github.com/stfc/aiida-mlip).

