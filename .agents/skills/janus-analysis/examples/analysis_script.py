#!/usr/bin/env python3
"""Example post-processing script for MD trajectory analysis using janus_core."""

from ase.io import read
from janus_core.processing.correlator import compute_vaf
from janus_core.processing.observables import compute_msd, compute_rdf


def main():
    traj_file = "md-traj.xyz"
    print(f"Reading trajectory from {traj_file}...")
    try:
        trajectory = read(traj_file, index=":")
    except Exception as e:
        print(f"Note: Trajectory file {traj_file} not found: {e}")
        return

    # 1. Radial Distribution Function
    print("Computing Radial Distribution Function (RDF)...")
    r_bins, gr = compute_rdf(
        trajectory,
        elements=["Na", "Cl"],
        rmax=10.0,
        nbins=100,
    )
    print(f"RDF computed over {len(r_bins)} bins.")

    # 2. Mean Squared Displacement
    print("Computing Mean Squared Displacement (MSD)...")
    msd = compute_msd(trajectory)
    print(f"MSD computed for {len(msd)} frames.")

    # 3. Velocity Autocorrelation Function
    print("Computing Velocity Autocorrelation Function (VAF)...")
    time_lags, vaf = compute_vaf(
        trajectory=traj_file,
        dt_fs=1.0,
        max_lag_fs=2000.0,
    )
    print("Post-processing analysis finished successfully.")


if __name__ == "__main__":
    main()

