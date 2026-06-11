"""
Run UMA on fixed DFT geometries.

"""

import argparse
import numpy as np
from ase.io import read


def main():
    parser = argparse.ArgumentParser(description="Evaluate UMA on ASE xyz frames.")
    parser.add_argument("ads_xyz", help="Input ASE extxyz file with DFT frames.")
    parser.add_argument("out_npz", help="Output npz file.")
    parser.add_argument("--model", default="uma-s-1p1", help="UMA model name.")
    parser.add_argument("--task", default="oc20", help="UMA task name, e.g. oc20, omat, omol.")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda.")
    args = parser.parse_args()

    print("[UMA] Loading DFT frames")
    frames = read(args.ads_xyz, index=":")

    if not isinstance(frames, list):
        frames = [frames]

    print(f"[UMA] Frames loaded: {len(frames)}")
    print(f"[UMA] Model: {args.model}")
    print(f"[UMA] Task: {args.task}")
    print(f"[UMA] Device: {args.device}")

    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:
        raise ImportError(
            "FAIRChem/UMA is not available in this environment. "
            "Run this script with the conda environment where fairchem-core is installed."
        ) from exc

    predictor = pretrained_mlip.get_predict_unit(
        args.model,
        device=args.device
    )

    calc = FAIRChemCalculator(
        predictor,
        task_name=args.task
    )

    energies = []
    forces = []

    for i, atoms in enumerate(frames):
        print(f"[UMA] Evaluating frame {i + 1}/{len(frames)}")
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        force = atoms.get_forces()

        energies.append(float(energy))
        forces.append(np.asarray(force, dtype=float))

    np.savez(
        args.out_npz,
        energies=np.array(energies),
        forces=np.stack(forces),
    )

    print(f"[UMA] Saved results to {args.out_npz}")


if __name__ == "__main__":
    main()