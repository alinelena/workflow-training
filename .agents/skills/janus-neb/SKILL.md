---
name: janus-neb
description: >-
  Calculate minimum energy pathways (MEP), transition states, and activation energy barriers
  using the Nudged Elastic Band (NEB) method with Climbing Image (CI-NEB) in STFC janus-core.
  Use this skill when investigating reaction mechanisms, ion diffusion pathways, or barrier heights.
---

# Janus-Core: Nudged Elastic Band (NEB)

Locate transition states, reaction energy barriers, and atomic migration paths connecting two known local energy minima.

## Key Procedures

### 1. Basic Climbing Image NEB (CI-NEB)

Finds the saddle point along the reaction path and computes forward and backward activation energy barriers:
```bash
janus neb \
  --arch mace_mp \
  --init-struct initial_state.cif \
  --final-struct final_state.cif \
  --n-images 5 \
  --climb \
  --fmax 0.05 \
  --device cpu \
  --no-tracker
```

### 2. IDPP Interpolation

Janus-Core uses Image-Dependent Pair Potential (IDPP) by default to avoid unphysical atomic overlaps when interpolating intermediate images:
```bash
janus neb \
  --arch mace_mp \
  --init-struct LiFePO4_start.cif \
  --final-struct LiFePO4_end.cif \
  --n-images 7 \
  --interpolator idpp \
  --climb \
  --fmax 0.02 \
  --steps 400 \
  --no-tracker
```

### 3. Using Pre-Generated Images

If you have already generated and inspected intermediate band images in an XYZ file:
```bash
janus neb \
  --arch mace_mp \
  --neb-structs band_images.xyz \
  --climb \
  --fmax 0.03 \
  --no-tracker
```

### 4. Python API Example

```python
from ase.io import read
from janus_core.calculations.neb import NEB

initial = read("initial_state.cif")
final = read("final_state.cif")

neb_calc = NEB(
    init_struct=initial,
    final_struct=final,
    n_images=5,
    arch="mace_mp",
    climb=True,
    interpolator="idpp",
    fmax=0.03,
)
neb_calc.calc()
print("NEB calculation completed.")
```

See [options.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-neb/references/options.md) for full options reference.

