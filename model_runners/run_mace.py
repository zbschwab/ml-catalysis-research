"""
Run MACE on fixed DFT geometries.

"""

import argparse
import numpy as np
from ase.io import read
from mace.calculators import mace_mp


def main():
    parser = argparse.ArgumentParser(description="Evaluate MACE on ASE xyz frames.")
    parser.add_argument("ads_xyz", help="Input ASE extxyz file with DFT frames.")
    parser.add_argument("out_npz", help="Output npz file.")
    parser.add_argument("--slab", default=None, help="Optional clean slab xyz.")
    parser.add_argument("--gas", default=None, help="Optional gas-phase xyz.")
    parser.add_argument("--model", default="medium", help="MACE-MP model size.")
    args = parser.parse_args()

    print("[MACE] Loading DFT frames")
    frames = read(args.ads_xyz, index=":")

    if not isinstance(frames, list):
        frames = [frames]

    print(f"[MACE] Frames loaded: {len(frames)}")
    print(f"[MACE] Model: {args.model}")

    calc = mace_mp(model=args.model, dispersion=False, default_dtype="float64")

    energies = []
    forces = []

    for i, atoms in enumerate(frames):
        print(f"[MACE] Evaluating frame {i + 1}/{len(frames)}")
        atoms.calc = calc

        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())

    np.savez(
        args.out_npz,
        energies=np.array(energies),
        forces=np.stack(forces),
    )

    print(f"[MACE] Saved results to {args.out_npz}")


if __name__ == "__main__":
    main()