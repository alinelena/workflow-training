---
name: janus-eos-elasticity
description: >-
  Calculate Equation of State (EOS) and full 6x6 elasticity stiffness tensors (C_ij)
  for crystalline materials using STFC janus-core.
  Computes bulk modulus, shear modulus, Young's modulus, and Poisson's ratio.
  Use this skill when evaluating mechanical stability, compressibility, and elasticity of crystals.
---

# Janus-Core: Equation of State & Elasticity

Evaluate the mechanical and thermodynamic properties of crystalline materials under volumetric and shear deformations.

## Key Procedures

### 1. Equation of State (EOS) Calculation

Fits energy-volume ($E-V$) curves to compute equilibrium volume ($V_0$), equilibrium energy ($E_0$), and isothermal bulk modulus ($B_0$):
```bash
janus eos \
  --arch mace_mp \
  --struct structure.cif \
  --min-volume 0.95 \
  --max-volume 1.05 \
  --n-volumes 7 \
  --eos-type birchmurnaghan \
  --device cpu \
  --no-tracker
```

Supported EOS types: `birchmurnaghan`, `murnaghan`, `vinet`, `pouriertarantola`, `antonschmidt`.

### 2. Elasticity Stiffness Tensor ($C_{ij}$)

Applies small normal and shear strains, calculates MLIP stress responses, and fits the 6x6 stiffness matrix $C_{ij}$:
```bash
janus elasticity \
  --arch mace_mp \
  --struct structure.cif \
  --normal-magnitude 0.01 \
  --shear-magnitude 0.06 \
  --n-strains 4 \
  --device cpu \
  --no-tracker
```

This outputs:
- Full 6x6 $C_{ij}$ stiffness tensor (in GPa)
- Voigt, Reuss, and Hill (VRH) averages for:
  - Bulk Modulus ($K_{VRH}$)
  - Shear Modulus ($G_{VRH}$)
  - Young's Modulus ($E$)
  - Poisson's Ratio ($\nu$)
  - Cauchy pressure ($C_{12} - C_{44}$) and Pugh's ratio ($K/G$) for ductility evaluation.

### 3. Pre-Optimization Before Mechanical Tests

To avoid spurious non-zero stress, pass `--minimize`:
```bash
janus elasticity \
  --arch mace_mp \
  --struct structure.cif \
  --minimize \
  --no-tracker
```

### 4. Python API Example

```python
from ase.io import read
from janus_core.calculations.eos import EoS
from janus_core.calculations.elasticity import Elasticity

atoms = read("structure.cif")

# 1. Equation of State
eos_calc = EoS(
    struct=atoms,
    arch="mace_mp",
    min_volume=0.95,
    max_volume=1.05,
    n_volumes=7,
    eos_type="birchmurnaghan",
)
eos_res = eos_calc.calc()
print(f"Bulk Modulus (EOS): {eos_res['bulk_modulus']:.2f} GPa")

# 2. Elasticity
el_calc = Elasticity(
    struct=atoms,
    arch="mace_mp",
    normal_magnitude=0.01,
    shear_magnitude=0.06,
)
el_res = el_calc.calc()
print("Stiffness Matrix C_ij (GPa):\n", el_res["elastic_tensor"])
```

See [eos_options.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-eos-elasticity/references/eos_options.md) and [elasticity_options.md](https://github.com/alinelena/ai-rules/blob/main/janus-core/skills/janus-eos-elasticity/references/elasticity_options.md) for full options.

