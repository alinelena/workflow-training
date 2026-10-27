#!/usr/bin/env python3
"""
Structure Inspector Tool for atomic structures (CIF, XYZ, POSCAR, PDB).
Extracts composition, unit cell, volume, density, and symmetry details.
Used by agents to verify atomic configurations before simulation.
"""

import argparse
import json
from pathlib import Path
import sys

try:
    from ase.io import read
except ImportError:
    read = None


def inspect_structure(file_path: str, as_json: bool = False):
    p = Path(file_path)
    if not p.exists():
        print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if read is None:
        print(
            "Error: 'ase' (Atomic Simulation Environment) is not installed in this Python environment.\n"
            "Please run with the janus micromamba environment:\n"
            "  micromamba run -n janus python janus-core/tools/structure_inspector.py " + file_path + "\n"
            "or run with uv:\n"
            "  uv run --with ase python janus-core/tools/structure_inspector.py " + file_path,
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        atoms = read(str(p))
    except Exception as e:
        print(f"Error reading structure '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    cell = atoms.get_cell()
    lengths = cell.lengths().tolist()
    angles = cell.angles().tolist()
    volume = float(atoms.get_volume())
    formula = str(atoms.get_chemical_formula())
    n_atoms = len(atoms)
    pbc = [bool(x) for x in atoms.get_pbc()]
    elements = sorted(list(set(atoms.get_chemical_symbols())))

    info = {
        "file": str(p.resolve()),
        "formula": formula,
        "n_atoms": n_atoms,
        "elements": elements,
        "pbc": pbc,
        "cell_lengths_angstrom": [round(x, 4) for x in lengths],
        "cell_angles_deg": [round(x, 2) for x in angles],
        "volume_angstrom3": round(volume, 4),
    }

    if as_json:
        print(json.dumps(info, indent=2))
    else:
        print("=" * 50)
        print(f"Structure File: {p.name}")
        print(f"Chemical Formula: {formula}")
        print(f"Total Atoms: {n_atoms}")
        print(f"Elements Present: {', '.join(elements)}")
        print(f"Periodic Boundary (x, y, z): {pbc}")
        print(f"Lattice Parameters (a, b, c): {info['cell_lengths_angstrom']} Å")
        print(f"Lattice Angles (alpha, beta, gamma): {info['cell_angles_deg']}°")
        print(f"Cell Volume: {info['volume_angstrom3']} Å³")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect atomic structure coordinates, cell parameters, and composition."
    )
    parser.add_argument("structure", help="Path to atomic structure file (CIF, XYZ, etc.)")
    parser.add_argument("--json", action="store_true", help="Output details as JSON")
    args = parser.parse_args()

    inspect_structure(args.structure, as_json=args.json)


if __name__ == "__main__":
    main()
