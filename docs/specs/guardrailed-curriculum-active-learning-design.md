# DESIGN DOC: Guardrailed Curriculum Active Learning WorkGraph (EpochMesh + Necropsy-Rewind)

## 1. Why
- **Problem**: Active learning molecular dynamics (MD) with MACE foundation models routinely discovers soft-core "extrapolation holes", leading to atomic clashes ($r_{ij} < 1.0\text{ Å}$) or unit cell explosions. Submitting these unphysical crash frames to Quantum ESPRESSO causes electronic charge sloshing, negative charge densities, and non-convergent Davidson loops that deadlock automated AiiDA workflows. In addition, fine-tuning MACE without strict regularization causes catastrophic forgetting of 0 K harmonic wells and bulk modulus.
- **Reference**: Grounded in the formal PRD at `docs/prds/guardrailed-curriculum-active-learning-workgraph-2026-10-27.md` and conceptual brief at `docs/eg-brainstorms/mace-qe-active-learning-workgraph-2026-10-27.md`.

## 2. Scope
### In-Scope:
- `pyproject.toml`: Modern packaging manifest configuring `hatchling` build backend, top-level package namespace `workflow_training`, dependencies (`aiida-core`, `aiida-workgraph`, `aiida-mlip`, `janus-core`, `ase`, `numpy`, `scipy`), and test dependencies (`pytest`, `pytest-cov`, `pytest-asyncio`).
- `src/workflow_training/active_learning/ring_buffer.py`: Pre-allocated circular FIFO ring buffer storing positions, velocities, forces, cell, and scalar metrics with zero per-step memory allocations (< 15 µs overhead) and full query accessors.
- `src/workflow_training/active_learning/watchdogs.py`: Multi-modal real-time physical watchdogs (atomic clashes $r_{ij} < 0.70 R_{\text{cov}}$, force spikes $\|\mathbf{F}_i\| > 25\text{ eV/Å}$, volume dilation $\Delta V / V_0 > 20\%$, cell condition number $\kappa(\mathbf{h}) < 0.15$) implemented as an ASE observer hook.
- `src/workflow_training/active_learning/necropsy.py`: Post-mortem causal analyzer executing reverse baseline Z-score inflection search ($3.5\sigma$ over MAD baseline), discarding the corpse frame, and extracting the pre-collapse cusp $t^*$.
- `src/workflow_training/active_learning/stages/stage0_cold_start.py`: Cold-start sub-workgraph builder generating EOS isotropic strains ($\pm 10\%$, 7 points) and harmonic rattles (0.02 Å to 0.05 Å on equilibrium), populating the permanent 20% Anchor-Replay Reservoir Group and initial `SinglefileData` dataset.
- `src/workflow_training/active_learning/stages/stage1_thermal.py`: Thermal exploration sub-workgraph builder executing guarded NPT MD across an ascending $(T, P)$ ladder with single-model tri-zone force filtering.
- `src/workflow_training/active_learning/dataset_collector.py`: AiiDA calcfunction accepting dynamic keyword inputs (`**gathered_dft`) to preserve full provenance, filtering out unconverged DFT runs, quarantining electronic failures into `QuarantineData` (`orm.Dict` + Group), blending 80% new data with 20% anchor replay, and writing extended `.xyz` (`orm.SinglefileData`).
- `src/workflow_training/active_learning/invariance_gate.py`: 5-stage pre-deployment gate validating ground-state residual forces ($F_{\max} \le 0.025\text{ eV/Å}$), bulk modulus retention ($|\Delta B_0| / B_0^{\text{DFT}} \le 8\%$, reconciled with PRD FR-15), acoustic $\Gamma$-phonon stability ($\omega^2 \ge -10^{-3}\text{ THz}^2$), $\mathrm{SO}(3)$ equivariance, and 2 ps NVE/NPT burn-in.
- `src/workflow_training/active_learning/training_config.py`: AiiDA calcfunction generating `JanusConfigfile` for `mlip.train` with dynamic loss weights and headless `tracker: false`.
- `src/workflow_training/active_learning/workgraph.py`: Top-level AiiDA WorkGraph integrating Stage 0, While loop zone, dynamic `workgraph.map_zone` (with concurrency throttle `max_number_jobs`), `select_next_model` context guard, `mlip.train` execution, and context hot-swapping.
- `tests/`: Comprehensive unit and integration test suite with offline mock fixtures:
  - `tests/conftest.py`
  - `tests/test_ring_buffer.py`
  - `tests/test_watchdogs.py`
  - `tests/test_necropsy.py`
  - `tests/test_invariance_gate.py`
  - `tests/test_eos_cold_start.py`
  - `tests/test_workgraph_builder.py`
  - `tests/test_al_mock_e2e.py`

### Out-of-Scope:
- Multi-fidelity DFT hierarchies (e.g. low-cutoff pre-screening).
- Non-Quantum ESPRESSO electronic structure codes.
- Multi-head ensemble committee training (deferred to v2 per PRD).

## 3. Package Structure & Namespace
```
workflow-training/
├── pyproject.toml              # Hatchling packaging setup, dependencies, pytest/ruff config
├── tools/
│   └── structure_inspector.py  # Structure inspector & coordinate validator
├── src/
│   └── workflow_training/
│       └── active_learning/
│           ├── __init__.py
│           ├── ring_buffer.py          # Zero-allocation NumPy circular trajectory buffer
│           ├── watchdogs.py            # ASE observer watchdog hooks (clash, force, volume, skew)
│           ├── necropsy.py             # Reverse Z-score inflection search & cusp extractor
│           ├── dataset_collector.py    # DFT validator, quarantine, & 20% replay mixer (**kwargs provenance)
│           ├── training_config.py      # JanusConfigfile builder for mlip.train
│           ├── invariance_gate.py      # 5-stage pre-deployment physical sanity check
│           ├── stages/
│           │   ├── __init__.py
│           │   ├── stage0_cold_start.py# EOS & harmonic rattle sub-workgraph builder
│           │   └── stage1_thermal.py   # Guarded NPT MD runner & candidate extractor
│           └── workgraph.py            # Top-level WorkGraph generator
└── tests/
    ├── conftest.py                     # Mock code fixtures, pseudo family, clean profile
    ├── test_ring_buffer.py
    ├── test_watchdogs.py
    ├── test_necropsy.py
    ├── test_invariance_gate.py
    ├── test_eos_cold_start.py
    ├── test_workgraph_builder.py
    └── test_al_mock_e2e.py
```

## 4. Interfaces & Detailed Data Model

### 4.1 Top-Level WorkGraph Generator (`workgraph.py`)
```python
def build_guardrailed_al_workgraph(
    structure: orm.StructureData,
    model: ModelData,
    pw_code: orm.InstalledCode,
    janus_code: orm.InstalledCode,
    pseudo_family: orm.Str,
    curriculum_config: orm.Dict,
    qe_parameters: orm.Dict = None,
    max_number_jobs: int = 16,
) -> WorkGraph:
    """
    Builds the top-level Active Learning WorkGraph.
    
    1. Pre-validates structure with tools/structure_inspector.py.
    2. Runs Stage 0 (EOS + harmonic rattles) to seed initial SinglefileData and 20% Anchor-Replay Group.
    3. Initializes WorkGraph context (model, structure, dataset, weights, replay_ratio, temp_idx).
    4. Enters While loop zone running Stage 1:
       Guarded NPT MD -> Map Zone QE -> Collector -> Config Prep -> Retrain -> Invariance Gate -> Model Selector.
    """
```

#### Schema for `qe_parameters: orm.Dict`:
```python
{
    "ecutwfc": 60.0,            # Plane wave energy cutoff (Ry)
    "ecutrho": 480.0,           # Charge density cutoff (Ry)
    "kpoints_distance": 0.03,   # Reciprocal mesh spacing (1/Å)
    "conv_thr": 1.0e-8,         # SCF convergence threshold (Ry)
    "smearing": "cold",         # Smearing scheme
    "degauss": 0.02,            # Smearing width (Ry)
    "tstress": True,            # Compute stress tensor
    "tprnfor": True             # Compute atomic forces
}
```

#### Schema for `curriculum_config: orm.Dict`:
```python
{
    "temperatures": [300.0, 600.0, 900.0],  # Ascending temperature ladder (K)
    "pressure_bar": 0.0,                     # Ambient isobaric pressure
    "md_steps_per_epoch": 10000,             # Number of MD steps per exploration
    "timestep_fs": 1.0,                      # MD timestep (fs)
    "max_epochs": 10,                        # Ceiling to prevent runaway loops
    "tri_zone": {
        "f_low": 0.8,                        # Lower force threshold (eV/Å): below is redundant equilibrium
        "f_high": 8.0,                       # Upper force threshold (eV/Å): above is unstable/cusp candidate
        "max_candidates_per_epoch": 20       # Cap on DFT jobs per epoch
    },
    "necropsy": {
        "clash_ratio": 0.70,                 # Fractional sum of covalent radii (r_ij < 0.70 (R_i + R_j))
        "force_max": 25.0,                   # Force explosion ceiling (eV/Å)
        "vol_dilation": 0.20,                # Maximum relative volume change |V-V0|/V0
        "cell_skew_min": 0.15,               # Minimum condition number of cell matrix
        "safe_ratio": 0.85                   # Minimum separation for extracted cusp
    },
    "training": {
        "energy_weight": 1.0,
        "forces_weight": 20.0,
        "stress_weight": 5.0,
        "learning_rate": 1e-4,
        "max_num_epochs": 50,
        "batch_size": 8,
        "anchor_replay_ratio": 0.20          # Guaranteed 20% replay of Stage 0
    }
}
```

### 4.2 Trajectory Ring Buffer (`ring_buffer.py`)
```python
class TrajectoryRingBuffer:
    def __init__(self, capacity: int, n_atoms: int):
        self.capacity: int
        self.n_atoms: int
        self.cursor: int
        self.count: int

    def append(self, step: int, atoms: Atoms, forces: np.ndarray, epot: float, ekin: float) -> None: ...
    def reconstruct_atoms(self, slot: int, atomic_numbers: np.ndarray) -> Atoms: ...
    def get_chronological_slots(self) -> np.ndarray: ...
    def get_force_history(self) -> np.ndarray: ...       # 1D array of max atomic force per frame
    def get_energy_history(self) -> tuple[np.ndarray, np.ndarray]: ... # (epot, ekin)
    def get_volume_history(self) -> np.ndarray: ...      # 1D array of cell volumes
    def get_step_history(self) -> np.ndarray: ...        # 1D array of step integers
    def __len__(self) -> int: ...
```

### 4.3 Real-Time Watchdogs (`watchdogs.py`)
```python
class SimulationWatchdogTriggered(RuntimeError):
    def __init__(self, reason: str, step: int, metrics: dict): ...

class NecropsyWatchdog:
    def __init__(
        self,
        atoms: Atoms,
        dyn: Dynamics,
        buffer_capacity: int = 100,
        f_max_threshold: float = 25.0,
        clash_ratio_threshold: float = 0.70,
        vol_dilation_threshold: float = 0.20,
        cell_skew_threshold: float = 0.15,
    ): ...
    def step(self) -> None:
        """
        Called by ASE dyn.attach() at every MD step.
        Monitors:
        1. Atomic clash: r_ij < clash_ratio_threshold * (R_cov_i + R_cov_j)
        2. Force spike: max ||F_i|| > f_max_threshold
        3. Volume explosion: |V - V0| / V0 > vol_dilation_threshold
        4. Cell shear collapse: cond(cell) < cell_skew_threshold
        """
    def _eval_clash(self) -> tuple[bool, float, tuple[int, int]]: ...
    def _trigger_halt(self, reason: str, metrics: dict) -> None: ...
```

### 4.4 Post-Mortem Inflection Cusp Extraction (`necropsy.py`)

#### Reverse Z-Score Mathematical Formulation:
$$
\begin{aligned}
\widetilde{F} &= \operatorname{median}(\{F_{\max}(\tau)\}_{\tau=t_0}^{t_{\text{crash}}-5}) \\
\text{MAD} &= 1.4826 \times \operatorname{median}(\{|F_{\max}(\tau) - \widetilde{F}|\}_{\tau=t_0}^{t_{\text{crash}}-5}) \\
Z(t) &= \frac{F_{\max}(t) - \widetilde{F}}{\text{MAD} + 10^{-6}}
\end{aligned}
$$

The pre-collapse cusp frame $t^*$ is extracted by scanning backward from $t_{\text{crash}}-1$:
$$
t^* = \max \left\{ t \le t_{\text{crash}}-1 \;\middle|\; Z(t) \le 3.5 \;\land\; \min_{i < j} r_{ij}(t) \ge 0.85 (R_{\text{cov}, i} + R_{\text{cov}, j}) \right\}
$$

```python
def find_inflection_cusp(
    buffer: TrajectoryRingBuffer,
    atomic_numbers: np.ndarray,
    safe_ratio: float = 0.85,
    baseline_sigma_mult: float = 3.5,
) -> dict:
    """
    Executes reverse Z-score inflection search across the ring buffer.
    Strictly discards the terminal crashed frame (slot -1).
    
    Z-Score Math:
    - Baseline computed from pre-crash history window (t_1 to t - 5):
      median_F = np.median(F_history)
      mad_F = 1.4826 * np.median(np.abs(F_history - median_F))
      Z_t = (F_t - median_F) / (mad_F + 1e-6)
    - If buffer length < 5 (early trajectory crash), falls back to t=0 ground state.
    - Scans chronologically backward from slot -2:
      Locates first frame where Z_t <= baseline_sigma_mult and min r_ij >= safe_ratio * (R_i + R_j).
    - If candidate still breaches safe_ratio, continues backtracking up to 10 frames earlier;
      if still unsafe, marks is_safe_for_dft = False.
      
    Returns dict:
    {
        "cusp_atoms": Atoms,          # Valid frontier structure right before divergence
        "approach_atoms": Atoms | None,# Structure 5 frames prior for context
        "cusp_step": int,
        "cusp_max_force": float,
        "discarded_crash_step": int,
        "reason": str,
        "is_safe_for_dft": bool       # Verified min r_ij >= safe_ratio * (R_i + R_j)
    }
    """
```

### 4.5 Guarded MD & Stage Builders (`stages/`)

#### Execution Boundary Architecture:
- `janus md` CLI executable does not support runtime ASE in-memory observer hooks.
- Therefore, Guarded MD is implemented as a standalone, headless Python runner `run_guarded_md` in `src/workflow_training/active_learning/stages/stage1_thermal.py`.
- In AiiDA WorkGraph, `run_guarded_md` is wrapped as an AiiDA `@task.calcfunction` (or `@task.pythonjob` on HPC).
- **Internal Exception Shield**: `SimulationWatchdogTriggered` is caught internally by `run_guarded_md`. It **never** bubbles an unhandled exception to AiiDA; instead, it executes `find_inflection_cusp`, extracts the safe pre-crash configuration, and returns a clean dictionary:
  `{"status": "crashed|completed", "cusp_structure": orm.StructureData | None, "candidate_structures": list[orm.StructureData], "metrics": orm.Dict}`.

```python
# stage0_cold_start.py
def build_stage0_subgraph(
    structure: orm.StructureData,
    pw_code: orm.InstalledCode,
    pseudo_family: orm.Str,
    qe_parameters: orm.Dict,
    strain_range: list[float] = [-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10],
    n_rattles: int = 4,
    rattle_std: float = 0.03,
) -> WorkGraph:
    """
    Sub-workgraph:
    1. Generates 7 strained configurations (±10% isotropic volume variation).
    2. Applies harmonic rattles (0.02-0.05 Å) to the equilibrium (0.0 strain) structure.
    3. Evaluates all configurations via Quantum ESPRESSO PwBaseWorkChain.
    4. Packages results into initial SinglefileData and registers to Anchor-Replay Group.
    """

# stage1_thermal.py
def run_guarded_md_task(
    structure: orm.StructureData,
    model: ModelData,
    temperature: orm.Float,
    pressure_bar: orm.Float,
    md_steps: orm.Int,
    timestep_fs: orm.Float,
    tri_zone_config: orm.Dict,
    necropsy_config: orm.Dict,
) -> dict:
    """
    Executes guarded NPT MD with NecropsyWatchdog.
    Catches watchdog triggers internally and extracts inflection cusp.
    Applies single-model tri-zone force filtering:
    - Informative frames (f_low <= max ||F_i|| <= f_high) are selected.
    - Caps candidates at max_candidates_per_epoch via temporal uniform sampling.
    Returns:
    {
        "status": orm.Str("completed" | "crashed"),
        "candidates": list[orm.StructureData],
        "cusp_structure": orm.StructureData (or None),
        "metrics": orm.Dict
    }
    """
```

### 4.6 Dataset Collector & Quarantine (`dataset_collector.py`)
```python
@task.calcfunction
def collect_filter_and_accumulate_dataset(
    previous_dataset: orm.SinglefileData,
    anchor_group_label: orm.Str,
    epoch_idx: orm.Int,
    anchor_replay_ratio: orm.Float = orm.Float(0.20),
    **gathered_dft: orm.Dict,
) -> dict:
    """
    Accepts dynamic keyword arguments (**gathered_dft) from mapped DFT tasks to maintain strict AiiDA provenance.
    1. Audits gathered PwBaseWorkChain results.
    2. Quarantines non-converged SCF calculations into QuarantineData (orm.Dict) and logs to Group.
    3. Pre-validates atomic distances (r_min >= 0.85 R_cov) using structure_inspector logic.
    4. Loads verified anchor configurations from the permanent Stage 0 Group.
    5. Blends (1 - anchor_replay_ratio) new valid DFT frames with anchor_replay_ratio anchor frames.
       If n_valid == 0 (e.g. 100% electronic failure), returns previous_dataset unmodified.
    6. Returns:
       {
           "updated_dataset": orm.SinglefileData (.xyz),
           "quarantine_report": orm.Dict,
           "n_valid": orm.Int,
           "n_quarantined": orm.Int
       }
    """
```

### 4.7 MACE Training Config Builder (`training_config.py`)
```python
@task.calcfunction
def prepare_mace_training_config(
    dataset_file: orm.SinglefileData,
    base_model: ModelData,
    training_params: orm.Dict,
) -> JanusConfigfile:
    """
    Generates the JanusConfigfile node required by CalculationFactory("mlip.train").
    Injects:
    - dataset path and validation split
    - loss weights: lambda_E, lambda_F, lambda_S
    - lr, batch_size, max_num_epochs
    - tracker: false (Headless execution rule)
    """
```

### 4.8 5-Stage Invariance Gate (`invariance_gate.py`)
```python
@task.calcfunction
def evaluate_invariance_gate(
    candidate_model: ModelData,
    baseline_structure: orm.StructureData,
    baseline_b0: orm.Float,
    device: orm.Str = orm.Str("cpu"),
) -> orm.Dict:
    """
    Evaluates candidate checkpoint against the 5 physical invariance criteria:
    1. Ground-state force residual: F_max <= 0.025 eV/Å, F_rms <= 0.010 eV/Å.
    2. EOS & Bulk Modulus: Birch-Murnaghan fit across V/V0 in [0.94, 1.06],
       |ΔB0| / B0_DFT <= 8% (PRD FR-15), B0 > 0.
    3. Acoustic Gamma-Phonon Stability: Finite displacement Hessian has no imaginary modes (ω² >= -1e-3 THz²).
    4. SO(3) Equivariance: Random rigid rotation has energy drift < 1e-5 eV, force error < 1e-4 eV/Å.
    5. 2 ps NVE & NPT Burn-In: NVE energy drift < 1e-4 eV/(atom*ps), NPT mean volume within ±2.5%.
    
    Returns orm.Dict({
        "passed": bool,
        "residual_force": float,
        "delta_b0_percent": float,
        "phonons_stable": bool,
        "equivariance_error": float,
        "burn_in_drift": float,
        "details": str
    })
    """

@task.calcfunction
def select_next_model(
    candidate_model: ModelData,
    fallback_model: ModelData,
    gate_result: orm.Dict,
) -> ModelData:
    """
    Context guard: Hot-swaps to candidate_model ONLY if gate_result['passed'] is True.
    Otherwise, preserves fallback_model.
    """
```

## 5. MLIP Calculator & Simulation Parameters
- **MLIP Architecture**: `mace_mp` foundation model fine-tuned via `mlip.train`.
- **Loss Formulation**:
$$
\mathcal{L} = \lambda_E \mathcal{L}_{\text{Huber}}(E, E_{\text{DFT}}) + \lambda_F \mathcal{L}_{\text{Huber}}(\mathbf{F}, \mathbf{F}_{\text{DFT}}) + \lambda_S \mathcal{L}_{\text{Huber}}(\boldsymbol{\sigma}, \boldsymbol{\sigma}_{\text{DFT}})
$$
- **Loss Weights**: $\lambda_E = 1.0$ (Huber), $\lambda_F = 20.0$ (Huber), $\lambda_S = 5.0$ (Huber on Cauchy stress).
- **Optimization**: AdamW ($\text{lr} = 10^{-4}$), `ReduceLROnPlateau` (factor 0.6, patience 6, max 50 epochs).
- **DFT Parameters**: Quantum ESPRESSO `PwBaseWorkChain` with CG diagonalization, cold smearing `degauss = 0.02 Ry`, `tstress = True`, `tprnfor = True`.
- **Cell Filter**: Always `FrechetCellFilter` for 0 K relaxations.
- **Headless Flag**: Explicitly inject `tracker: false` into all `janus-core` MD kwargs and `JanusConfigfile` nodes.

## 6. Execution Environment & Provenance
- **Environment**: Micromamba `janus` (`/opt/micromamba/envs/janus/bin/python`).
- **AiiDA Nodes**: Inputs/outputs as `StructureData`, `ModelData`, `JanusConfigfile`, `SinglefileData` (.xyz), and `Group`.
- **While Zone Predicate**:
  ```python
  @task.calcfunction
  def evaluate_loop_condition(epoch: orm.Int, is_converged: orm.Bool, max_epochs: orm.Int) -> orm.Bool:
      return orm.Bool((not is_converged.value) and (epoch.value < max_epochs.value))
  ```
- **Context State Schema (`wg.context`)**:
  - `current_model`: Active `ModelData` checkpoint (protected by `select_next_model`).
  - `current_structure`: Active `StructureData`.
  - `dataset_file`: Cumulative `SinglefileData` (.xyz).
  - `epoch_idx`: Current iteration integer.
  - `temp_idx`: Index into `curriculum_config["temperatures"]`.
  - `loss_weights`: Current training loss weights dict (boosted 1.5× on gate rejection).
  - `anchor_replay_ratio`: Current replay ratio (boosted to 0.30 on gate rejection).
  - `retry_count`: Consecutive gate failures counter.

## 7. Failure Modes & Recovery Workflows
1. **Unphysical Crash Frame**: Discarded; rewound to cusp $t^*$ with $\min r_{ij} \ge 0.85 R_{\text{cov}}$ guarantee.
2. **Invariance Gate Rejection**: If `evaluate_invariance_gate` returns `passed=False`, candidate checkpoint is rejected; `select_next_model` keeps `fallback_model`. The WorkGraph retries with loss weights boosted by 1.5× and anchor replay ratio expanded to 30%.
3. **Zero Candidate Frames in Stable MD**: If MD completes without tripping watchdogs and no frames fall within $[f_{\text{low}}, f_{\text{high}}]$, the curriculum increments `temp_idx` to advance to $T_{\text{next}}$. If already at $T_{\max}$, the loop terminates with `is_converged = True`.
4. **Empty Dataset from 100% DFT Quarantine**: If all candidate frames in a batch fail electronic SCF convergence or clash checks, `dataset_collector` leaves the training dataset unchanged, and the MD trajectory is restarted from the previous epoch's stable checkpoint with a randomized velocity seed.
5. **HPC Queue Overload**: Enforce `wg.max_number_jobs = 16` on the WorkGraph engine.
6. **Hard Ceiling**: Terminate at `max_epochs = 10` to prevent runaway compute.

## 8. Verification Criteria & Test Plan
- **Pre-Commit Linting**: `uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check src tests`
- **Unit Tests**:
  - `tests/conftest.py`: Defines mock fixtures (`mock_pw_code`, `mock_janus_code`, `mock_pseudo_family`, `clean_profile`) enabling offline unit testing without live cluster access.
  - `tests/test_ring_buffer.py`: Verifies buffer capacity, append latency (< 15 µs), and chronological reconstruction.
  - `tests/test_watchdogs.py`: Verifies clash ratio detection, force explosion trigger, volume dilation breach, and cell skew condition number.
  - `tests/test_necropsy.py`: Verifies reverse Z-score inflection search correctly extracts pre-collapse cusp and discards corpse frame; handles short buffers ($t < 5$).
  - `tests/test_invariance_gate.py`: Verifies 5-stage gate checks (forces, EOS fit, phonons with $-10^{-3}\text{ THz}^2$ tolerance, equivariance, burn-in) and `select_next_model` hot-swap guard.
  - `tests/test_eos_cold_start.py`: Verifies Stage 0 deformation series generation and harmonic rattle seeding.
  - `tests/test_workgraph_builder.py`: Verifies WorkGraph DAG construction, task port wiring, context mapping, and concurrency limits.
  - `tests/test_al_mock_e2e.py`: End-to-end integration test with mocked DFT/MLIP tasks verifying loop execution, quarantine routing, and model hot-swapping under clean AiiDA test profile.
- **Test Runner**: Full test pass via `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest tests/`.
