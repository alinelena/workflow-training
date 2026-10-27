# Guardrailed Curriculum Active Learning Examples

This directory contains standalone, runnable examples demonstrating the core components of the Guardrailed Curriculum Active Learning WorkGraph (`EpochMesh` + `Necropsy-Rewind`) for MACE foundation model fine-tuning and Quantum ESPRESSO in `aiida-workgraph`.

---

## 1. Overview of Examples

| Script | Purpose | Focus Area |
|---|---|---|
| [`01_guardrailed_workgraph.py`](01_guardrailed_workgraph.py) | Full WorkGraph construction & DAG wiring | Demonstrates how an end-user configures Diamond Silicon, curriculum ladders, watchdogs, and Quantum ESPRESSO parameters to build the top-level WorkGraph. |
| [`02_watchdogs_and_necropsy.py`](02_watchdogs_and_necropsy.py) | In-flight watchdogs & Necropsy-Rewind | Runs a standalone ASE molecular dynamics trajectory, catches real-time atomic clashes, and executes the reverse Z-score inflection search to extract pre-collapse cusp $t^*$. |
| [`03_offline_quarantine_and_gate.py`](03_offline_quarantine_and_gate.py) | Quarantine, replay mixing & 5-stage gate | Audits a batch of DFT calculations, isolates electronic SCF failures into `QuarantineData`, mixes a 20% anchor replay quota, and evaluates the 5-stage physical invariance pre-deployment gate. |

### Visual Architecture Diagram

A publication-quality SVG diagram of the dataset building and fine-tuning pipeline is saved at [`dataset_building_workflow.svg`](dataset_building_workflow.svg) (and [`docs/images/dataset_building_workflow.svg`](../docs/images/dataset_building_workflow.svg)).

---

## 2. Requirements & Environment

- **Python**: $\ge$ 3.12 (or `>= 3.12`)
- **Core Dependencies**: `janus-core >= 0.9.8`, `aiida-workgraph >= 0.4.0`, `aiida-mlip >= 0.1.0`, `ase`, `numpy`, `scipy`.
- **Headless Execution Rule**: All calculations and training configurations automatically inject `tracker: false` to disable CodeCarbon emission overhead.

---

## 3. Running the Examples

Run any example directly using Python:

```bash
# 1. Inspect top-level WorkGraph DAG
python examples/01_guardrailed_workgraph.py

# 2. Trigger real-time watchdogs and extract inflection cusp
python examples/02_watchdogs_and_necropsy.py

# 3. Test quarantine routing and the 5-stage physical invariance gate
python examples/03_offline_quarantine_and_gate.py
```

