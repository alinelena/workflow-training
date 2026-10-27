# Product Requirements Document: Guardrailed Curriculum Active Learning WorkGraph (EpochMesh + Necropsy-Rewind)

**Status**: Approved Concept $\to$ Formal PRD  
**Author**: Antigravity (Pair Programming Agent)  
**Date**: 2026-10-27  
**Artifact Path**: `docs/prds/guardrailed-curriculum-active-learning-workgraph-2026-10-27.md`  
**Execution Environment**: Micromamba `janus` (`/opt/micromamba/envs/janus/bin/python`), `uv`  
**Target Plugins**: `aiida-workgraph`, `aiida-mlip`, `aiida-quantumespresso`, STFC `janus-core`  

---

## 1. Executive Summary

### 1.1 One-Sentence Elevator Pitch
An automated, fault-tolerant active learning WorkGraph in `aiida-workgraph` that iteratively fine-tunes MACE foundation models across an ascending thermodynamic curriculum (0 K EOS $\to$ isobaric NPT MD) using Quantum ESPRESSO DFT labeling, protected by an in-flight **Necropsy-Rewind** stability guardrail that prevents unphysical crash structures from ever reaching the electronic structure solver.

### 1.2 Target User & Use Case
* **Target User**: Computational materials scientists, solid-state chemists, and atomistic simulation engineers seeking to adapt general-purpose MACE foundation models (e.g. `mace_mp`) to specific crystalline systems, high-temperature phase diagrams, or high-pressure regimes without manual active-learning script orchestration.
* **Core Use Case**: Given a pristine crystal structure (`StructureData`) and a pre-trained MACE checkpoint (`ModelData`), autonomously explore configuration space, detect model extrapolation failures, trigger Quantum ESPRESSO single-points only for high-value frontier configurations, and fine-tune the model until 100 ps NPT MD runs stably across target $(T, P)$ conditions.

### 1.3 Success Criteria
* **DFT Completion Reliability**: $\ge 98\%$ electronic SCF convergence rate for all candidate structures submitted to Quantum ESPRESSO `PwBaseWorkChain` (zero deadlocked WorkGraph processes due to atomic overlap or unphysical cell distortion).
* **Zero Catastrophic Forgetting**: The fine-tuned MACE model must preserve 0 K harmonic mechanical stability:
  * Bulk modulus variation $|\Delta B_0| / B_0^{\text{DFT}} < 6\%$.
  * Ground-state force residual $F_{\max} \le 0.025\text{ eV/\AA}$.
  * Zero imaginary acoustic phonon frequencies at $\Gamma$.
* **MD Thermal Stability**: The final model must complete a continuous 100 ps NPT trajectory at the target temperature without triggering stability watchdogs or unphysical volume expansion/collapse.
* **Orchestration Fault Isolation**: Failed DFT calculations or aborted MD runs are quarantined without crashing the parent WorkGraph; verified training points are immutably grouped with complete AiiDA provenance.

### 1.4 Scope Summary
* **In-Scope (v1 / MVP)**:
  - Stage 0 (Cold Start): Equation of State (EOS) isotropic volume deformations ($\pm 10\%$) + harmonic rattles.
  - Stage 1 (Thermal Exploration): Ascending isobaric $(T, P)$ NPT MD ramps (e.g., $300\text{ K} \to 600\text{ K} \to 1000\text{ K}$).
  - In-memory circular trajectory ring-buffer with real-time multi-modal watchdogs (atomic clashes, force explosions, volume dilation).
  - Reverse Z-score inflection cusp extraction ($t^*$) discarding unphysical corpse frames.
  - Quantum ESPRESSO dynamic batch fan-out via `workgraph.map_zone` with concurrency cap (`max_number_jobs`).
  - Automated DFT result collector, quarantine validator, and AiiDA dataset `Group` accumulator.
  - MACE fine-tuning via `mlip.train` with 20% Anchor-Replay Reservoir, Huber loss, stress loss weighting, and model checkpoint hot-swapping.
  - 5-stage pre-deployment validation gate before checkpoint promotion.
* **Out-of-Scope (v1)**:
  - Multi-fidelity DFT hierarchies (e.g., fast PBEsol pre-screening leading into HSE06).
  - Unconstrained melting / liquid-gas phase diagram discovery.
  - Non-Quantum ESPRESSO DFT codes (e.g., VASP, CP2K, CASTEP).

---

## 2. Problem Statement & User Journey

### 2.1 The Pain Point
Active learning for machine-learned interatomic potentials is plagued by the **"Garbage-In, Deadlock-Out" DFT Trap**:
1. When a foundation MLIP explores high temperatures or strains, it eventually falls into an "extrapolation hole"—an unconstrained soft-core region where repulsive forces are underestimated.
2. Under thermal inertia, atoms fuse ($r_{ij} < 1.0\text{ \AA}$) or unit cells explosively dilate.
3. Naive active learning workflows sample the crashed frame and submit it to DFT.
4. Quantum ESPRESSO encounters overlapping atomic pseudopotential cores, resulting in extreme charge sloshing, negative charge densities, or non-convergent Davidson diagonalization.
5. In automated workflow managers, an unhandled DFT failure stalls the pipeline, requiring manual database surgery to rescue the calculation.

Furthermore, fine-tuning MACE on specialized active-learning data often induces **catastrophic forgetting**: the model gains accuracy on high-energy distorted states but loses its 0 K harmonic well, causing stable crystals to spontaneously amorphize at ambient temperatures.

### 2.2 The "Magic Moment"
The user submits a single `StructureData` and `ModelData` to AiiDA. The workflow autonomously executes cold-start volume sampling, initiates NPT molecular dynamics, intercepts an incipient unphysical trajectory at 800 K, safely rewinds 70 fs to the physical cusp preceding the collapse, computes Quantum ESPRESSO forces/stress with guaranteed convergence, fine-tunes MACE with a baseline replay buffer, verifies bulk modulus retention, and resumes the MD ramp until complete stability is attained.

### 2.3 User Persona
* **Dr. Elena Vance (Computational Materials Scientist)**:
  - *Goal*: Run 100 ns NPT simulations of a newly synthesized halide perovskite at 600 K.
  - *Frustration*: Foundation MACE-MP explodes after 4 ps due to an unphysical iodine-iodine clash. Running active learning manually requires babysitting dozens of SLURM scripts, copying XYZ files, handling QE SCF crashes, and manually retraining PyTorch checkpoints.
  - *Needs*: A hands-off, robust AiiDA WorkGraph that runs on the cluster, guarantees DFT jobs succeed, and outputs an experimentally reliable, fine-tuned potential.

### 2.4 End-to-End User Journey
```
[User Input]
  │ StructureData (pristine crystal)
  │ ModelData (MACE-MP checkpoint)
  │ Curriculum Config (T_max = 900 K, P = 0 bar)
  ▼
[Stage 0: Cold Start Sub-WorkGraph]
  │ Deform cell (EOS ±10%) + Harmonic rattles
  │ Run batch QE pw.x singlepoints
  │ Initialize Active Learning Dataset Group
  │ Initial MACE Fine-Tune & Validate
  ▼
[Stage 1: Curriculum Thermal Exploration Loop]
  │ Loop: T in [300 K, 600 K, 900 K]
  │ While trajectory trips watchdogs:
  │    ┌───────────────────────────────────────────────┐
  │    │ 1. Run Guardrailed NPT MD (janus-core / ASE) │
  │    │    - Watchdog catches clash or force spike    │
  │    │ 2. Discard crash frame; Rewind to cusp t*    │
  │    │ 3. Dispatch t* to QE PwBaseWorkChain          │
  │    │ 4. Accumulate into Dataset Group (20% Replay) │
  │    │ 5. Fine-Tune MACE (mlip.train with stress)   │
  │    │ 6. Run 5-Stage Invariance Validation Gate     │
  │    │ 7. Hot-swap ModelData -> Next iteration      │
  │    └───────────────────────────────────────────────┘
  ▼
[Convergence Gate]
  │ 100 ps NPT MD runs at T_max without triggering rewinds
  ▼
[Output Artifacts]
  │ Final Certified ModelData
  │ Complete Lineage Group (all DFT evaluations)
  │ Stability & Invariance Audit Report
```

---

## 3. Codebase Grounding & Technical Conventions

### 3.1 Existing Surfaces & Analogues
* **High-Throughput WorkGraph Builder**:  
  [`aiida-mlip/skills/aiida-mlip-workflows/examples/submit_ht_workgraph.py`](file:///home/drFaustroll/playground/ml/workflow-training/aiida-mlip/skills/aiida-mlip-workflows/examples/submit_ht_workgraph.py): Model for constructing WorkGraphs with concurrency management (`wg.max_number_jobs`).
* **MACE Training Task (`mlip.train`)**:  
  [`aiida-mlip/skills/aiida-mlip-train/examples/submit_train.py`](file:///home/drFaustroll/playground/ml/workflow-training/aiida-mlip/skills/aiida-mlip-train/examples/submit_train.py): Invokes `CalculationFactory("mlip.train")` using `foundation_model`, `fine_tune=Bool(True)`, and `JanusConfigfile`.
* **MD Exploration Task (`mlip.md`)**:  
  [`aiida-mlip/skills/aiida-mlip-md/examples/submit_md.py`](file:///home/drFaustroll/playground/ml/workflow-training/aiida-mlip/skills/aiida-mlip-md/examples/submit_md.py): Invokes `CalculationFactory("mlip.md")` with Martyna-Tobias-Klein NPT dynamics.
* **Equation of State Deformations**:  
  [`janus-core/skills/janus-eos-elasticity/SKILL.md`](file:///home/drFaustroll/playground/ml/workflow-training/janus-core/skills/janus-eos-elasticity/SKILL.md): `from janus_core.calculations.eos import EoS`.
* **Pre-Flight Geometric Inspector**:  
  [`tools/structure_inspector.py`](file:///home/drFaustroll/playground/ml/workflow-training/tools/structure_inspector.py): Base utility for auditing atomic distance matrices and PBC consistency.

### 3.2 Architectural & Project Constraints (from `AGENTS.md`)
* **Environment**: Always execute in `/opt/micromamba/envs/janus/bin/python` via `uv`.
* **Headless Simulation**: All simulation parameter dictionaries and `JanusConfigfile` nodes must pass `tracker: false` (`--no-tracker`) to avoid CodeCarbon emission overhead.
* **Cell Relaxations**: When performing geometry optimizations, always set `filter: Str("FrechetCellFilter")`.
* **AiiDA ORM**: Inputs/outputs must strictly use native AiiDA nodes (`ModelData`, `StructureData`, `JanusConfigfile`, `SinglefileData`, `Dict`, `Group`).
* **Cluster Assumptions**: Both `janus@localhost` (or cluster janus) and `pw` (`aiida-quantumespresso`) are operational on the target system.

---

## 4. Detailed Requirements

### 4.1 Functional Requirements

#### Stage 0: Cold-Start & Baseline Initialization
* **FR-01 (Priority: P0)**: The system SHALL generate an Equation of State (EOS) deformation series covering $\pm 10\%$ volumetric strain (minimum 7 points) and 4 random harmonic displacement supercells ($0.02 - 0.05\text{ \AA}$ displacement) from the initial structure.
* **FR-02 (Priority: P0)**: The system SHALL evaluate all Stage 0 configurations with Quantum ESPRESSO (`PwBaseWorkChain`), storing energies, forces, and Cauchy stress tensors.
* **FR-03 (Priority: P0)**: The system SHALL register all Stage 0 DFT calculations in an immutable AiiDA `Group` labeled `al_anchor_reservoir_<uuid>`, serving as the permanent anti-forgetting replay baseline.
* **FR-04 (Priority: P0)**: The system SHALL perform an initial MACE fine-tuning run via `mlip.train` on the Stage 0 dataset to align energy offsets and stress calibrations before starting MD.

#### Necropsy-Rewind Stability Engine
* **FR-05 (Priority: P0)**: The system SHALL monitor molecular dynamics trajectories in real time using an in-memory zero-allocation circular FIFO ring buffer storing the latest $W \ge 100$ frames (positions, velocities, forces, cell, energies).
* **FR-06 (Priority: P0)**: The system SHALL evaluate three orthogonal safety watchdogs at every MD integration step:
  - *Atomic Clash Watchdog*: $\exists (i, j) : r_{ij} < 0.70 \times (R_{\text{cov}, i} + R_{\text{cov}, j})$.
  - *Force Explosion Watchdog*: $\max_i \|\mathbf{F}_i\| > 25.0\text{ eV/\AA}$ or presence of `NaN`/`Inf`.
  - *Volume Dilation Watchdog*: $|V(t) - V_0| / V_0 > 0.20$ or cell condition number $< 0.15$.
* **FR-07 (Priority: P0)**: Upon breach of any watchdog, the system SHALL immediately halt MD dynamics, discard the crashed terminal configuration ($t_{\text{crash}}$), and execute a reverse baseline Z-score inflection search across the ring buffer to locate the pre-collapse cusp $t^*$.
* **FR-08 (Priority: P0)**: The extracted cusp configuration $t^*$ SHALL satisfy a strict pre-flight physical safety check ($\min r_{ij} \ge 0.85 (R_{\text{cov}, i} + R_{\text{cov}, j})$) before being forwarded to Quantum ESPRESSO.

#### Active Learning Filtering & Quantum ESPRESSO Labeling
* **FR-09 (Priority: P0)**: During stable MD trajectories (where watchdogs are not tripped), the system SHALL filter candidate frames using a tri-zone force variance / uncertainty rule ($\sigma_{\text{low}} \le \sigma_F \le \sigma_{\text{high}}$), discarding both redundant low-energy frames ($\sigma_F < \sigma_{\text{low}}$) and unphysical outliers ($\sigma_F > \sigma_{\text{high}}$).
* **FR-10 (Priority: P0)**: The system SHALL dispatch candidate structures dynamically to Quantum ESPRESSO `PwBaseWorkChain` using `workgraph.map_zone`, enforcing a user-configurable concurrency cap (`wg.max_number_jobs = 16`).
* **FR-11 (Priority: P0)**: The system SHALL automatically parameterize Quantum ESPRESSO for high electronic stability on strained frontier geometries (Conjugate Gradient diagonalization, adaptive mixing `mixing_beta = 0.20`, Marzari-Vanderbilt cold smearing `degauss = 0.02 Ry`, and full stress computation `tstress = True`).
* **FR-12 (Priority: P1)**: If a DFT calculation fails to converge within 100 iterations, the system SHALL quarantine the structure into a `QuarantineData` node and continue without deadlocking the WorkGraph.

#### MACE Fine-Tuning & Model Hot-Swapping
* **FR-13 (Priority: P0)**: The system SHALL append newly validated DFT frames to the training dataset while enforcing a strict **20% Anchor-Replay Reservoir quota** drawn from Stage 0 baseline structures.
* **FR-14 (Priority: P0)**: The system SHALL execute `mlip.train` with Huber loss, loss weights $\lambda_E = 1.0$, $\lambda_F = 20.0$, $\lambda_S = 5.0$, AdamW optimizer ($\text{lr} = 10^{-4}$), and `ReduceLROnPlateau` scheduler (patience 6 epochs, max 50 epochs).
* **FR-15 (Priority: P0)**: Before promoting a newly fine-tuned `ModelData` node, the system SHALL evaluate the 5-Stage Invariance Gate:
  1. Pristine 0 K residual force $F_{\max} \le 0.025\text{ eV/\AA}$.
  2. Bulk modulus deviation $|\Delta B_0| / B_0^{\text{DFT}} < 8\%$ with positive curvature ($B_0 > 0$).
  3. 2 ps NVE burn-in energy drift $< 10^{-4}\text{ eV/(atom}\cdot\text{ps})$.
* **FR-16 (Priority: P0)**: Upon passing the gate, the system SHALL hot-swap the output `ModelData` into the WorkGraph context (`wg.context['current_model']`), propagating the updated potential to the next curriculum temperature ramp.

### 4.2 Non-Functional Requirements
* **NFR-01 (Memory Overhead)**: The in-memory trajectory ring buffer SHALL consume $\le 10\text{ MB}$ of RAM for systems up to 1,000 atoms.
* **NFR-02 (Step Latency)**: Real-time watchdog evaluation in ASE/janus MD SHALL incur $\le 20\ \mu\text{s}$ per MD integration step ($< 2\%$ total simulation overhead on GPU).
* **NFR-03 (AiiDA Provenance)**: Every fine-tuned `ModelData`, candidate `StructureData`, and DFT output SHALL be permanently connected in the AiiDA provenance graph, enabling full retrospective reproduction.
* **NFR-04 (Fault Containment)**: Any unexpected crash in an individual sub-task SHALL fail soft, logging diagnostic telemetry to an AiiDA `Dict` node without terminating the top-level WorkGraph.

---

## 5. Proposed Architecture & WorkGraph Topologies

### 5.1 WorkGraph Control Flow Diagram
```
                     ┌────────────────────────────────────────────────────────┐
                     │                   WorkGraph Context                    │
                     │  - current_structure: StructureData                    │
                     │  - current_model: ModelData                            │
                     │  - dataset_group: Group                                │
                     │  - stage_idx: Int (0, 1, 2)                            │
                     │  - continue_learning: Bool                             │
                     └───────────────────────────┬────────────────────────────┘
                                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │ Stage 0: Cold Start   │
                                     │ - EOS (±10%) + Rattle │
                                     │ - Batch QE pw.x       │
                                     │ - Seed Anchor Group   │
                                     │ - Initial Train M₀    │
                                     └───────────┬───────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Dynamic While Zone: while continue_learning == True and epoch < max_epochs                      │
│                                                                                                 │
│  [1. Guardrailed MD Task (janus-core / ASE)]                                                    │
│      ├── Normal Completion ──► [Tri-Zone Force Variance Filter] ──► [Select Frames]             │
│      └── Watchdog Breach  ──► [Necropsy-Rewind: Cusp t*]         ──► [Select Cusp Frame]        │
│                                                                             │                   │
│                                                                             ▼                   │
│  [2. Dynamic Map Zone: PwBaseWorkChain (max_number_jobs = 16)] ◄────────────┘                   │
│      ├── Converged SCF ──► [Extract Energies, Forces, Stresses]                                 │
│      └── Diverged SCF  ──► [Route to QuarantineData (Skip)]                                     │
│                                     │                                                           │
│                                     ▼                                                           │
│  [3. Dataset Accumulator & Replay Mixer Task]                                                   │
│      └── Mix 80% New DFT Data + 20% Anchor-Replay Reservoir ──► SinglefileData (.xyz)           │
│                                     │                                                           │
│                                     ▼                                                           │
│  [4. MACE Fine-Tuning Task (mlip.train)]                                                        │
│      └── Inputs: foundation_model=current_model, dataset, weights(E=1, F=20, S=5)               │
│      └── Output: candidate_model (ModelData)                                                    │
│                                     │                                                           │
│                                     ▼                                                           │
│  [5. 5-Stage Invariance & Physical Sanity Gate]                                                 │
│      ├── PASS ──► Update Context: current_model = candidate_model; advance curriculum T        │
│      └── FAIL ──► Discard candidate; tighten loss weights; retry with expanded anchor set       │
│                                                                                                 │
│  [6. Convergence Evaluator]                                                                     │
│      └── If 100 ps NPT MD passes without watchdogs or high variance at T_max:                   │
│             continue_learning = False                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │ Final Certified Model │
                                     │ Provenance Export     │
                                     └───────────────────────┘
```

### 5.2 Component Inventory
| Component | Type | Responsibility |
| :--- | :--- | :--- |
| `curriculum_orchestrator` | `WorkGraph` | Top-level DAG orchestrator managing context, `While` loop, and stage transitions. |
| `eos_cold_start_subgraph` | `@task.graph_builder` | Generates volumetric/shear strains and harmonic displacements; triggers initial QE single-points. |
| `necropsy_md_task` | `@task.calcfunction` | Executes guarded MD with in-memory ring buffer; halts on watchdogs and performs reverse Z-score inflection rewind. |
| `dft_batch_map_zone` | `workgraph.map_zone` | Dynamically scatters candidate structures across cluster `PwBaseWorkChain` slots with `max_number_jobs`. |
| `dataset_collector_task` | `@task` | Validates DFT outputs, quenches SCF failures, mixes 20% anchor replay, and formats `.xyz` datasets. |
| `mace_finetune_task` | `CalculationFactory("mlip.train")` | Executes transfer learning on MACE foundation checkpoints with Huber loss and stress weights. |
| `invariance_gate_task` | `@task.calcfunction` | Validates bulk modulus $B_0$, 0 K residual forces, and 2 ps NVE/NPT burn-in before promoting model. |

### 5.3 Data Flow & Interfaces
```python
# Signature for top-level WorkGraph generator
def build_mace_qe_active_learning_workgraph(
    structure: orm.StructureData,
    model: ModelData,
    pw_code: orm.InstalledCode,
    janus_code: orm.InstalledCode,
    pseudo_family: orm.Str,
    curriculum_config: orm.Dict,
    max_number_jobs: int = 16,
) -> WorkGraph: ...
```

---

## 6. Edge Cases, Failure Modes & Mitigations

| Failure Mode / Edge Case | Direct Impact | Automated Mitigation | Detection Point |
| :--- | :--- | :--- | :--- |
| **Exploded frame sent to DFT** | Severe charge sloshing, negative density abort in `pw.x`, stalled WorkGraph. | **Necropsy-Rewind**: Discard terminal frame; rewind to pre-collapse cusp $t^*$; assert $\min r_{ij} \ge 0.85 R_{\text{cov}}$. | `necropsy_md_task` pre-flight check. |
| **Catastrophic forgetting of ground state** | 0 K crystal becomes mechanically unstable; negative bulk modulus $B_0 < 0$. | **Anchor-Replay Reservoir**: Enforce 20% representation of 0 K / EOS configurations in every training batch. | `invariance_gate_task` Birch-Murnaghan fit. |
| **NPT box explosion / volume runaway** | Simulation box dilates into vacuum while internal atomic forces appear moderate. | **Stress Loss Weighting**: Set $\lambda_S = 5.0$ in `mlip.train` to anchor cell virial response. | `necropsy_md_task` volume watchdog ($\Delta V/V_0 > 20\%$). |
| **Persistent electronic SCF divergence** | An unusual boundary configuration fails DFT even with adaptive mixing. | **Quarantine Vault**: Isolate unconverged structure into `QuarantineData`; skip frame without failing WorkGraph. | `dataset_collector_task` SCF accuracy audit. |
| **HPC cluster queue starvation** | Dynamic fan-out submits 200 concurrent QE jobs, blocking other users or exceeding SLURM limits. | **Concurrency Throttle**: Enforce `wg.max_number_jobs = 16` on the WorkGraph engine. | `aiida-workgraph` job scheduler queue. |
| **Perpetual rewind deadlock** | Material undergoes a true physical phase transition, perpetually tripping watchdogs. | **Curriculum Epoch Ceiling**: Terminate loop after `max_epochs = 10`, reporting phase transition flags to user. | `While` zone `max_iterations` counter. |

---

## 7. Rollout, Metrics & Verification Plan

### 7.1 Verification Gate (Automated Tests before Merge)
1. **Unit Test (`test_necropsy_buffer.py`)**:
   - Verify `TrajectoryRingBuffer` zero-allocation memory write speed ($< 20\ \mu\text{s}$).
   - Simulate a synthetic runaway trajectory; verify reverse Z-score correctly flags $t^*$ and discards the crash frame.
2. **Integration Test (`test_eos_cold_start.py`)**:
   - Run Stage 0 subgraph on a 2-atom Si cell; verify EOS deformations match Birch-Murnaghan fits and populate the AiiDA Anchor Group.
3. **End-to-End Smoke Test (`test_active_learning_e2e.py`)**:
   - Execute 2 full active learning loops on an Al supercell using lightweight pseudopotentials and a dummy/small MACE model in the `janus` environment with `--no-tracker`.
   - Verify `ModelData` hot-swapping and AiiDA provenance graph integrity.

### 7.2 Phased Rollout Milestones
* **Milestone 1 (Foundations & Necropsy Engine)**: Implement `tools/necropsy_rewind.py` with the ring buffer and ASE observer hook; validate against synthetic unphysical trajectories.
* **Milestone 2 (WorkGraph Subgraphs & Dynamic Fan-Out)**: Implement `eos_cold_start_subgraph` and `dft_batch_map_zone` with `max_number_jobs` concurrency control.
* **Milestone 3 (Loop Closure & Gate Integration)**: Implement the top-level `While` loop orchestrator, `dataset_collector_task` with 20% anchor replay, and the 5-Stage Invariance Gate.
* **Milestone 4 (End-to-End Validation & Documentation)**: Run a complete thermal curriculum active learning campaign ($300\text{ K} \to 800\text{ K}$) on a benchmark crystal system (e.g. Si or $\text{CsPbI}_3$).

---

## 8. Open Questions & Deferred Decisions

1. **Multi-Head vs. Full-Backbone Fine-Tuning**:
   - *Current Decision*: Full-backbone fine-tuning with a 20% Anchor-Replay Reservoir is the default for v1 to ensure maximal chemical adaptability.
   - *Deferred*: Native multi-head readout committee fine-tuning (`--multiheads_finetuning=True`) will be explored in v2 once upstream `janus-core` CLI exposes multi-head export endpoints.
2. **Dynamic Umbrella Biasing**:
   - *Deferred*: Adding automated harmonic bias potentials to deliberately push MD trajectories into high-pressure phase boundaries will be deferred to v2.
3. **Automated Pseudopotential Family Selection**:
   - *Current Decision*: User provides the SSSP string (e.g. `"SSSP/1.3/PBE/efficiency"`); automatic functional matching is deferred.

