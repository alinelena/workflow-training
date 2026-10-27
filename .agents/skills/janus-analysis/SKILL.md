---
name: janus-analysis
description: >-
  Post-process molecular dynamics trajectories and calculate structural and dynamic observables
  such as Radial Distribution Functions (RDF), Velocity Autocorrelation Functions (VAF),
  and Mean Squared Displacement (MSD) using STFC janus-core.
  Use this skill when analyzing simulation trajectories, liquid structures, diffusion, or vibrational spectra.
---

# Janus-Core: Trajectory Analysis & Observables

Analyze MD simulation trajectories to evaluate equilibrium structural order, transport properties, diffusion coefficients, and dynamical correlations using `janus_core.processing`.

## Key Procedures

### 1. Radial Distribution Function (RDF / $g(r)$)

Computes pair correlation functions to understand coordination shells and local liquid/amorphous structure:
```python
from ase.io import read
from janus_core.processing.observables import compute_rdf

# Read trajectory frames
trajectory = read("md-traj.xyz", index=":")

# Compute RDF for specified elements
r_bins, gr = compute_rdf(
    trajectory,
    elements=["Na", "Cl"],
    rmax=10.0,
    nbins=100,
)
print("RDF computed successfully.")
```

### 2. Velocity Autocorrelation Function (VAF / VACF)

Computes atomic velocity memory, dynamical relaxation, and vibrational spectra (via Fourier Transform of VAF):
```python
from janus_core.processing.correlator import compute_vaf

time_lags, vaf = compute_vaf(
    trajectory="md-traj.xyz",
    dt_fs=1.0,
    max_lag_fs=2000.0,
)
print("VAF calculated.")
```

### 3. Mean Squared Displacement (MSD) & Diffusion

Computes particle displacement over time to evaluate diffusion coefficients ($D = \lim_{t \to \infty} \frac{\langle |r(t) - r(0)|^2 \rangle}{6t}$):
```python
from ase.io import read
from janus_core.processing.observables import compute_msd

trajectory = read("md-traj.xyz", index=":")
msd_values = compute_msd(trajectory)
print(f"Calculated MSD across {len(msd_values)} frames.")
```

### 4. On-The-Fly Post-Processing During MD

Janus-Core can evaluate observables on-the-fly during molecular dynamics without saving massive trajectory files to disk, via `--post-process-kwargs` or YAML config:
```yaml
post_process_kwargs:
  rdf_compute: true
  rdf_rmax: 8.0
  rdf_nbins: 100
  rdf_elements: ["Li", "O"]
  vaf_compute: true
  vaf_fft: true
```

