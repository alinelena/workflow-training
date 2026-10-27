# AiiDA-MLIP Multi-Agent Execution Rules

When interacting with STFC `aiida-mlip` or submitting AiiDA calculations with MLIPs:

1. **Python Environment**:
   - Always run commands in the `janus` micromamba environment (`micromamba run -n janus python` or `micromamba run -n janus verdi ...`) or via `uv run --no-project python`.
   - Install packages and run tests using `uv`.

2. **AiiDA Daemon & Profile**:
   - Verify that the AiiDA profile and daemon are running (`verdi daemon status`) before submitting batch calculations.
   - Use `verdi presto` for quick profile and broker configuration if running locally.

3. **Installed Code Configuration**:
   - Ensure a `janus` installed code is registered in AiiDA (typically `janus@localhost`):
     ```bash
     verdi code show janus@localhost
     ```
   - Set up via `verdi code create core.code.installed` if not already present.

4. **Provenance & AiiDA Data Types**:
   - All inputs submitted to calculations (`SinglePoint`, `GeomOpt`, `MD`, `Phonons`, `Descriptors`, `Train`) must be AiiDA ORM data types (`StructureData`, `ModelData`, `JanusConfigfile`, `Dict`, `Str`, `Float`, `Int`, `Bool`).

5. **Headless Execution & Emission Tracking**:
   - When generating `JanusConfigfile` or configuration dictionaries, set `tracker: false` to disable CodeCarbon emission tracking overhead.

6. **Documentation & Live Reference**:
   - Query [official aiida-mlip documentation](https://stfc.github.io/aiida-mlip/) and [aiida-mlip GitHub repository](https://github.com/stfc/aiida-mlip).

