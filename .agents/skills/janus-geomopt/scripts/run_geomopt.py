#!/usr/bin/env python3
"""Helper script to run GeomOpt calculation via Janus Python API."""

import argparse
from pathlib import Path
import sys

from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt


def main():
    parser = argparse.ArgumentParser(description="Run Janus Geometry Optimization.")
    parser.add_argument("--struct", required=True, help="Input structure file")
    parser.add_argument("--arch", default="mace_mp", help="MLIP architecture (default: mace_mp)")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--fmax", type=float, default=0.01, help="Max force convergence tolerance (eV/Å)")
    parser.add_argument("--steps", type=int, default=500, help="Max optimization steps")
    parser.add_argument(
        "--filter",
        default=None,
        choices=["FrechetCellFilter", "ExpCellFilter", "UnitCellFilter"],
        help="Cell filter for cell relaxation",
    )
    parser.add_argument("--out", default=None, help="Output relaxed structure file")
    args = parser.parse_args()

    atoms = read(args.struct)
    opt = GeomOpt(
        struct=atoms,
        arch=args.arch,
        device=args.device,
        fmax=args.fmax,
        steps=args.steps,
        filter_func=args.filter,
    )
    opt.optimize()

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_f = abs(forces).max()

    print("Geometry Optimization Completed:")
    print(f"  Final Energy: {energy:.6f} eV")
    print(f"  Residual Max Force: {max_f:.6f} eV/Å")

    out_file = args.out if args.out else f"relaxed_{Path(args.struct).name}"
    write(out_file, atoms)
    print(f"  Saved relaxed structure to: {out_file}")


if __name__ == "__main__":
    main()

