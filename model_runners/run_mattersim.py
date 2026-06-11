"""
Run MatterSim on fixed DFT geometries.

"""

import argparse
import numpy as np
from ase.io import read


def main():
    parser = argparse.ArgumentParser(description="Evaluate MatterSim on ASE xyz frames.")
    parser.add_argument("ads_xyz", help="Input ASE extxyz file with DFT frames.")
    parser.add_argument("out_npz", help="Output npz file.")
    parser.add_argument("--slab", default=None, help="Optional clean slab xyz.")
    parser.add_argument("--gas", default=None, help="Optional gas-phase xyz.")
    args = parser.parse_args()

    print("[MatterSim] Loading DFT frames")
    frames = read(args.ads_xyz, index=":")

    if not isinstance(frames, list):
        frames = [frames]

    print(f"[MatterSim] Frames loaded: {len(frames)}")

    try:
        from mattersim.forcefield import MatterSimCalculator
    except ImportError as exc:
        raise ImportError(
            "MatterSim is not available in this environment. "
            "Run this script with the conda environment where MatterSim is installed."
        ) from exc

    calc = MatterSimCalculator()

    energies = []
    forces = []

    for i, atoms in enumerate(frames):

        print(f"[MatterSim] Evaluating frame {i+1}/{len(frames)}")

        atoms.calc = calc

        energy = atoms.get_potential_energy()

        try:
            force = atoms.get_forces()

            print("Force shape:", force.shape)
            print("Max force:", np.max(np.linalg.norm(force, axis=1)))

        except Exception as exc:
            print("Force extraction failed:", exc)
            force = None

        energies.append(energy)
        forces.append(force)

    forces = [np.asarray(f, dtype=float) for f in forces if f is not None]

    if any(f is None for f in forces):
        print("[MatterSim] Warning: some forces were not available.")
        np.savez(
            args.out_npz,
            energies=np.array(energies),
            forces=np.array(forces, dtype=object),
        )
    else:
        np.savez(
            args.out_npz,
            energies=np.array(energies),
            forces=np.stack(forces),
        )


if __name__ == "__main__":
    main()