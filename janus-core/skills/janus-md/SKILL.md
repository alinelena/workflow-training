---
name: janus-md
description: >-
  Run molecular dynamics (MD) simulations in NVE, NVT, or NPT ensembles using STFC janus-core.
  Supports Langevin and Nosé-Hoover thermostats, barostats, temperature ramping, and PLUMED metadynamics.
  Use this skill for running MD trajectories, sampling phase space, or calculating thermodynamic averages.
---

# Janus-Core: Molecular Dynamics (MD)

Simulate finite-temperature dynamics of atomic systems across standard statistical mechanical ensembles (microcanonical, canonical, isothermal-isobaric) with MLIP acceleration.

## Supported Ensembles

- `nve`: Microcanonical ensemble (constant $N, V, E$).
- `nvt`: Canonical ensemble with Langevin or Nosé-Hoover thermostat.
- `nvt-nh`: Nosé-Hoover thermostat canonical ensemble.
- `nvt-csvr`: Canonical sampling via velocity rescaling.
- `npt`: Isothermal-isobaric ensemble (fluctuating volume).
- `npt-mtk` / `npt-mtk-iso` / `npt-mtk-aniso`: Martyna-Tobias-Klein barostats.
- `nph`: Constant enthalpy and pressure ensemble.

---

## Key Procedures

### 1. NVT Simulation (Langevin Dynamics)

```bash
janus md \
  --ensemble nvt \
  --arch mace_mp \
  --struct structure.cif \
  --temp 300.0 \
  --timestep 1.0 \
  --steps 10000 \
  --friction 0.01 \
  --stats-every 10 \
  --traj-every 50 \
  --device cpu \
  --no-tracker
```

### 2. NPT Simulation with Martyna-Tobias-Klein Barostat

```bash
janus md \
  --ensemble npt-mtk \
  --arch mace_mp \
  --struct structure.cif \
  --temp 300.0 \
  --pressure 1.0 \
  --thermostat-time 50.0 \
  --barostat-time 75.0 \
  --timestep 1.0 \
  --steps 20000 \
  --stats-every 20 \
  --traj-every 100 \
  --no-tracker
```

### 3. Temperature Ramping (Annealing / Heating)

Simulate continuous heating or cooling:
```bash
janus md \
  --ensemble nvt \
  --arch mace_mp \
  --struct structure.cif \
  --temp-start 300.0 \
  --temp-end 600.0 \
  --temp-step 10.0 \
  --temp-time 100.0 \
  --timestep 1.0 \
  --steps 30000 \
  --no-tracker
```

### 4. Metadynamics via PLUMED

Janus-Core seamlessly connects with PLUMED for enhanced sampling and free energy calculations:
```bash
janus md \
  --ensemble nvt \
  --arch mace_mp \
  --struct system.xyz \
  --temp 300.0 \
  --steps 50000 \
  --plumed-input plumed.dat \
  --plumed-log plumed.log \
  --no-tracker
```

### 5. Python API Example

```python
from ase.io import read
from janus_core.calculations.md import NVT

atoms = read("structure.cif")
sim = NVT(
    struct=atoms,
    arch="mace_mp",
    device="cpu",
    timestep=1.0,  # fs
    temp=300.0,    # K
    steps=5000,
    friction=0.01,
    stats_every=10,
    traj_every=50,
)
sim.run()
```

See [options.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-md/references/options.md) for full options reference.

