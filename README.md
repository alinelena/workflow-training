# Workflow Training: Guardrailed Curriculum Active Learning

Guardrailed Curriculum Active Learning WorkGraph (`EpochMesh` + `Necropsy-Rewind`) for fine-tuning MACE Machine-Learned Interatomic Potentials (MLIPs) with Quantum ESPRESSO using STFC `janus-core` and `aiida-workgraph`.

## Key Capabilities

- **Stage 0 Cold-Start Anchor Generation**: 7-point Murnaghan EOS volume strains ($\pm 6\%$, step 2%) and Monte Carlo thermal rattles (0.01 Å to 0.08 Å RMSD) preventing catastrophic forgetting.
- **Stage 1 Guarded Thermal Sampling**: NPT molecular dynamics with real-time watchdogs for unphysical atomic clash ($r_{\min} < 0.6\,r_{\text{covalent}}$), explosive force spikes ($|\mathbf{F}| > 25\,\text{eV/\AA}$), volume dilation ($V/V_0 > 1.35$), and shear cell skew ($\kappa(\mathbf{h}) > 6.0$).
- **Necropsy-Rewind Inflection Cusp Detection**: Reverse Z-score anomaly scanning ($t^* = \operatorname{argmax}_t [Z(E_{\text{kin}}) + Z(F_{\max})]$) that identifies the exact inflection point preceding numerical blowup while strictly discarding the unphysical corpse frame.
- **Electronic Quarantine & 20% Anchor Replay**: Isolates SCF divergence, negative charge densities, and non-converged electronic failures into `QuarantineData`, blending 80% newly labeled valid frames with 20% frozen Stage 0 anchors.
- **5-Stage Invariance Gate**: Pre-deployment validation verifying bulk modulus convergence ($|\Delta B_0/B_0| \le 8\%$), force parity ($R^2 > 0.90$), $\mathrm{SO}(3)$ frame equivariance ($< 10^{-4}\,\text{eV}$), acoustic phonon stability ($\omega_{\min} \ge -0.05\,\text{THz}$), and 100-step burn-in MD.
- **Resource Throttling**: WorkGraph job concurrency capped via `max_number_jobs = 16`.

## Installation

```bash
micromamba run -n janus uv pip install -e ".[test]"
```

## Running the Examples

Ensure your environment is set up with Quantum ESPRESSO (`pw.x`) and `janus`:

```bash
# 1. Full WorkGraph DAG construction
micromamba run -n janus python examples/01_guardrailed_workgraph.py

# 2. Watchdogs & Necropsy-Rewind demonstration
micromamba run -n janus python examples/02_watchdogs_and_necropsy.py

# 3. Offline Quarantine & 5-Stage Invariance Gate
micromamba run -n janus python examples/03_offline_quarantine_and_gate.py

# 4. WorkGraph visualization (table and HTML viewer)
micromamba run -n janus python examples/04_show_workgraph.py
```

