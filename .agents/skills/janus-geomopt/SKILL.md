---
name: janus-geomopt
description: >-
  Perform atomic geometry optimization and unit cell relaxation using STFC janus-core.
  Supports algorithms like BFGS, LBFGS, FIRE, and cell filters like FrechetCellFilter and ExpCellFilter.
  Use this skill when relaxing atomic coordinates, optimizing crystal lattice vectors, or finding energy minima.
---

# Janus-Core: Geometry & Cell Optimization

Perform energy minimization to find equilibrium atomic configurations, relaxed bond lengths, and optimized crystal lattice dimensions.

## Key Procedures

### 1. Atomic Coordinate Optimization (Fixed Cell)

Minimizes residual forces on atoms while keeping the unit cell vectors fixed:
```bash
janus geomopt \
  --arch mace_mp \
  --struct initial.cif \
  --fmax 0.01 \
  --steps 500 \
  --device cpu \
  --no-tracker
```

### 2. Full Cell and Coordinate Relaxation

To optimize both atomic coordinates and periodic unit cell dimensions simultaneously, pass `--opt-cell-fully`:
```bash
janus geomopt \
  --arch mace_mp \
  --struct initial.cif \
  --opt-cell-fully \
  --filter FrechetCellFilter \
  --fmax 0.005 \
  --steps 1000 \
  --write-traj \
  --out relaxed.cif \
  --no-tracker
```

> [!IMPORTANT]
> Always prefer `--filter FrechetCellFilter` over older unit cell filters for stable metric convergence on periodic systems.

### 3. Optimization Under Hydrostatic Pressure

Optimize lattice parameters and atomic coordinates under external isotropic pressure (in GPa):
```bash
janus geomopt \
  --arch mace_mp \
  --struct initial.cif \
  --opt-cell-lengths \
  --pressure 2.5 \
  --fmax 0.01 \
  --no-tracker
```

### 4. Using Python API

```python
from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt

atoms = read("initial.cif")
opt = GeomOpt(
    struct=atoms,
    arch="mace_mp",
    device="cpu",
    fmax=0.01,
    steps=500,
    filter_func="FrechetCellFilter",
)
opt.optimize()

write("relaxed.cif", atoms)
print(f"Relaxed Potential Energy: {atoms.get_potential_energy():.6f} eV")
```

Or execute the provided helper script:
```bash
python janus-core/skills/janus-geomopt/scripts/run_geomopt.py \
  --struct initial.cif \
  --arch mace_mp \
  --filter FrechetCellFilter \
  --fmax 0.01 \
  --out relaxed.cif
```

See [options.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-geomopt/references/options.md) for full options reference.

