# CONCEPTS BRIEF: AiiDA Active Learning Workgraph for MACE & Quantum ESPRESSO

**Date**: 2026-10-27  
**Artifact**: Concepts Brief  
**Topic**: Automated MACE MLIP Active Learning & Fine-Tuning with Quantum ESPRESSO in AiiDA WorkGraph  

---

## 1. Summary
- **Seed**: Build an AiiDA workgraph using `aiida-mlip` and Quantum ESPRESSO to fine-tune a MACE MLIP, including data generation stages (EOS, NPT MD) and active learning around MD stability, given a user-supplied input structure and baseline model.
- **Goldfish run**: 4 subagents across 4 distinct lenses (*Workgraph Control Flow*, *Active Learning Criteria*, *Pre-Mortem & Robustness*, *Prior Art Workflow Patterns*).
- **Total raw concepts**: 12; deduplicated and synthesized to **5 deep concepts**.

---

## 2. Synthesized Concepts Across the Lenses

### Concept 1: The EpochMesh Curriculum (Battle-Tested Staged Orchestrator)
* **One-line pitch**: A staged macro-batching AiiDA WorkGraph that executes discrete, immutable epoch sub-workgraphs across an ascending thermodynamic ladder (EOS cold-start → isobaric NPT ramps) with tri-zone force filtering and an AiiDA `Group` blackboard.
* **Originating lenses**: Architecture + Prior Art (DP-GEN2 & AiiDA-WorkGraph patterns).
* **How it works**:
  - Rather than an unrolling dynamic cyclic graph, a top-level controller schedules immutable sub-workgraphs per epoch ($E_0, E_1, \dots, E_K$).
  - Stages progress systematically:
    - **Stage 0 (Cold Start)**: Isotropic/shear EOS volume deformations ($\pm 10\%$) and harmonic rattles.
    - **Stage 1 (Thermal Exploration)**: Ascending $(T, P)$ ladders (300 K → 600 K → 1000 K).
    - **Stage 2 (Boundary Stress)**: Defected or superheated supercells.
  - Candidate frames from `mlip.md` are tri-zone filtered: $\sigma_F < \sigma_{\text{low}}$ (accurate, discard); $\sigma_{\text{low}} \le \sigma_F \le \sigma_{\text{high}}$ (candidate pool); $\sigma_F > \sigma_{\text{high}}$ (unphysical explosion, discard).
  - Selected frames are labeled concurrently via `PwBaseWorkChain` and appended into an AiiDA dataset `Group`.
  - `mlip.train` warm-starts from the baseline checkpoint. When candidate selection rate falls below 1%, the workgraph advances to the next thermodynamic stage.
* **Strongest arguments for**:
  - Proven reliability across literature (DP-GEN); avoids complex dynamic DAG unrolling in `aiida-workgraph`.
  - Rock-solid fault isolation: if epoch 3 fails, epochs 0–2 remain valid, inspectable, and uncorrupted.
* **The hard questions**:
  - Long simulation latency: monolithic MD runs must finish before batch filtering, meaning a model that explodes at step 200 wastes the remaining 49,800 steps.
* **Concrete next step**: Define the WorkGraph `@task.graph` generator coordinating Stage 0 (EOS) and Stage 1 (NPT MD) subgraphs.

---

### Concept 2: The Necropsy-Rewind Guardrail (Causal Boundary Healer)
* **One-line pitch**: An anomaly-trapping MD driver that halts at the onset of stability breakdown, discards the unphysical crash frame, rewinds 50–100 fs to the pre-collapse inflection cusp, and repairs the repulsive barrier using ZBL priors and surgical QE labeling.
* **Originating lenses**: Active Learning Criteria + Pre-Mortem + Prior Art (FLARE on-the-fly).
* **How it works**:
  - `mlip.md` runs with a lightweight in-memory circular ring buffer caching the last 50 frames.
  - Multi-modal crash watchdogs detect breakdown: interatomic clash $r_{ij} < 0.70 (R_{\text{cov}, i} + R_{\text{cov}, j})$, force explosion $\|\mathbf{F}_i\| > 25\text{ eV/Å}$, or instantaneous volume dilation $> 15\%$.
  - The crash frame is **discarded** (it causes electronic charge sloshing and DFT SCF failure). The workflow scans backward along the ring buffer to the inflection point $t^*$ where force uncertainty first accelerated.
  - Frames at $t^* - \Delta t$ and $t^*$ are sent to Quantum ESPRESSO with automated self-healing SCF fallbacks (Conjugate Gradient diagonalization, adaptive smearing).
  - Synthetic Ziegler-Biersack-Littmark (ZBL) repulsive priors are automatically injected for sub-threshold distances.
* **Strongest arguments for**:
  - Direct cure for the #1 failure mode of DFT active learning: unphysical crash structures deadlocking Quantum ESPRESSO with SCF divergence.
  - Focuses DFT budget on the high-value boundary between physical stability and MLIP hallucination.
* **The hard questions**:
  - Requires custom callback/hook integration into `janus-core` MD to maintain the rolling buffer and early-stop cleanly.
* **Concrete next step**: Implement a prototype trajectory monitor script with a ring-buffer and geometric clash detector in `tools/`.

---

### Concept 3: The Multi-Head Representation Freeze (Ultra-Fast Linear Active Learning)
* **One-line pitch**: Eliminates deep-ensemble compute bottlenecks by freezing MACE's equivariant message-passing backbone and active-learning only an ensemble of scalar readout heads with a baseline replay reservoir, enabling 2-minute GPU retraining per cycle.
* **Originating lenses**: Prior Art (CamML Multi-Head MACE) + Pre-Mortem (Catastrophic Forgetting).
* **How it works**:
  - Loads foundation checkpoint (`MACE-MP-0`) and attaches $K=4$ or $K=8$ randomly seeded readout heads.
  - Single-pass inference during MD evaluates epistemic force variance across the heads with $< 5\%$ compute overhead compared to a single model (no 4× deep ensemble penalty).
  - Every retraining batch enforces an **Anchor-Replay Reservoir**: 30% of the training batch is permanently drawn from the foundation baseline dataset (equilibrium 0 K, standard EOS curves).
  - `mlip.train` updates only the linear/readout parameters, executing in 1–3 minutes on a single GPU.
  - Candidate checkpoints must pass an invariance gate (zero imaginary phonon frequencies at $\Gamma$, bulk modulus $B_0$ within 5% of baseline) before deployment.
* **Strongest arguments for**:
  - Eliminates catastrophic forgetting of foundation properties (negative-energy sinkholes, elastic collapse).
  - Turns retraining from a multi-hour bottleneck into a sub-5-minute AiiDA workgraph task.
* **The hard questions**:
  - If the user's material exhibits chemical bonding radically different from the pre-training distribution, a frozen backbone cannot learn new equivariant features.
* **Concrete next step**: Formulate the multi-head fine-tuning configuration flags for `janus-core` / `aiida-mlip`.

---

### Concept 4: Latent CUR-Leverage & Virial Drift Auditor (Thermodynamic Invariant Selector)
* **One-line pitch**: Dual-criterion active query selection combining unsupervised CUR matrix leverage on MACE penultimate invariant descriptors with extended Hamiltonian drift and isothermal compressibility auditing.
* **Originating lenses**: Active Learning Criteria + Workgraph Control Flow.
* **How it works**:
  - Structures passing thermal MD are featurized using `janus-descriptors` into invariant penultimate-layer embeddings $\mathbf{h}_i^{(L=0)}$.
  - Low-rank CUR matrix decomposition extracts statistical leverage scores $\pi_j$, selecting configurations spanning orthogonal, unexplored geometric subspaces.
  - Concurrently, NPT trajectories are audited for non-physical stress phenomena: extended Hamiltonian drift $\frac{d\mathcal{H}_{\text{ext}}}{dt} \neq 0$ and isothermal compressibility discrepancy $|\beta_T^{\text{MD}} - 1/B_0^{\text{EOS}}| > \epsilon$.
  - Candidates at the inflection extrema of breathing/shear oscillations are selected and labeled with full Quantum ESPRESSO stress tensors (`tstress=.true.`).
  - `mlip.train` is dynamically parameterized with higher loss weights on stress ($w_{\text{stress}}$).
* **Strongest arguments for**:
  - MACE fine-tuning failures in NPT are notoriously driven by inaccurate stress tensors that lead to unphysical volume drift while atomic forces appear small. This criterion directly heals mechanical and thermodynamic flaws.
  - Guarantees zero collinearity/redundancy among structures submitted to QE.
* **The hard questions**:
  - Computing full invariant descriptors and SVD/CUR leverage across thousands of trajectory frames can become a CPU/memory bottleneck inside AiiDA.
* **Concrete next step**: Prototype a standalone Python task utilizing `janus-descriptors` to perform CUR column selection on test trajectory frames.

---

### Concept 5: The PulseTrain Actor-Critic Stream (Asynchronous Non-Blocking Pipeline)
* **One-line pitch**: Decouples fast continuous GPU MD exploration from slow MPI CPU Quantum ESPRESSO evaluation through an asynchronous actor-critic queue with quorum-triggered checkpoint hot-swapping.
* **How it works**:
  - The workgraph runs three decoupled lanes:
    1. *Explorer Track*: Continuously dispatches MD trajectories across varying $(T, P)$ points.
    2. *Labeler Pool*: Dynamically consumes structures from an in-flight candidate queue, running `PwBaseWorkChain` up to a concurrency cap (e.g., max 10 concurrent jobs).
    3. *Trainer Worker*: An `@task.monitor` task triggers GPU `mlip.train` whenever a quorum ($N \ge 30$) of new validated DFT points accumulates.
  - Updated `ModelData` is hot-swapped into new MD jobs without stopping running workflows.
* **Strongest arguments for**:
  - Maximum cluster utilization: GPUs never idle waiting for slow CPU DFT queues, and CPU cores never idle waiting for PyTorch training.
* **The hard questions**:
  - Highest engineering complexity in AiiDA; risk of non-deterministic provenance graphs and "stale-model exploration runaway."
* **Concrete next step**: Prototype an `aiida-workgraph` monitor task with dynamic task dispatching.

---

## 3. Cross-Cutting Tensions

1. **Tight Synchronous Locking vs. Asynchronous Decoupling**:
   - *Synchronous (EpochMesh)* guarantees exact, deterministic AiiDA provenance and clean fault isolation, but leaves GPU/CPU resources idle between phases.
   - *Asynchronous (PulseTrain)* maximizes HPC throughput, but introduces non-deterministic queues and complex race conditions in AiiDA.
2. **Full-Backbone Fine-Tuning vs. Multi-Head Representation Freeze**:
   - *Full-Backbone* allows learning completely new chemistry, but risks catastrophic forgetting and takes hours per training cycle.
   - *Frozen Backbone (Multi-Head)* retrains in 2 minutes and preserves baseline stability, but cannot adapt if the target phase space requires new equivariant representations.
3. **Equilibrium Sampling vs. Adversarial Boundary Healing**:
   - Sampling typical MD frames improves force accuracy on known basins.
   - Adversarial boundary healing (Necropsy-Rewind + ZBL priors) prevents catastrophic MD explosions, which is the primary mode of failure in NPT.

---

## 4. The Elephant's Synthesis

- **Most Viable Direction for this Repo**: **Hybrid of Concept 1 (EpochMesh Curriculum) + Concept 2 (Necropsy-Rewind Guardrail)**.
  - Building discrete, staged epoch subgraphs (EOS → NPT) inside `aiida-workgraph` is 100% compliant with existing `aiida-mlip` capabilities.
  - Adding a **trajectory rewind + geometric sanitizer** before `PwBaseWorkChain` is non-negotiable: without it, unphysical crash frames will inevitably cause Quantum ESPRESSO SCF divergence and stall your active learning loop.
- **The Sleeper Pick**: **Concept 3 (Multi-Head Representation Freeze)**.
  - Freezing the message-passing core and fine-tuning only the readout heads with an anchor-replay buffer makes the retraining step trivial (minutes instead of hours) and protects against the model forgetting 0 K harmonic stability.
- **The "Don't Do This" Pick**: **Unconstrained Lockstep While-Loops without Geometric Sanitization**.
  - A naive loop that simply takes the crashed frame from an exploded MD run and throws it at `pw.x` will fail in production on day one due to DFT electronic charge sloshing.

---

## 5. Top 3 Ranked Picks

1. **The Guardrailed Curriculum Loop (EpochMesh + Necropsy-Rewind)**:
   - *Reasoning*: Couples a rock-solid, stage-by-stage thermodynamic workflow (EOS → NPT) with pre-collapse trajectory rewinding and geometric pre-filtering, guaranteeing 99%+ DFT completion rates without deadlocking AiiDA.
2. **The Fast Multi-Head Anchor-Replay Loop**:
   - *Reasoning*: Drastically slashes compute time by using single-pass multi-head uncertainty and frozen-core transfer learning with baseline replay, preventing catastrophic forgetting.
3. **The Virial Stress-Corrected NPT Loop**:
   - *Reasoning*: Directly targets the true root cause of NPT instability (stress tensor error and unphysical volume expansion/collapse) rather than merely looking at atomic forces.

