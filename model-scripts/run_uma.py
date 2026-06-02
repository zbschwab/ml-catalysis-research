"""
run_mattersim.py — run in mlip-mattersim environment
conda activate mlip-mattersim
"""

import sys
import os
import numpy as np
from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# print("hello from uma!")  # check connection

slab_ads_frames = read(sys.argv[1], index=":")
slab = read(sys.argv[2])
gas = read(sys.argv[3])

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", device="cpu"
)  # change to cuda if available

calc = FAIRChemCalculator(predictor, task_name="oc20", seed=None)

for atoms in slab_ads_frames:
    atoms.calc = calc

slab.calc = calc
gas.calc = calc

energies_list = []
forces_list = []

for atoms in slab_ads_frames:
    energies_list.append(atoms.get_potential_energy())
    forces_list.append(atoms.get_forces())

# print("saving to:", os.path.abspath(sys.argv[4]))
# print("cwd:", os.getcwd())

np.savez(
    sys.argv[4],
    mlip_energies=np.array(energies_list),
    mlip_forces=np.array(forces_list),
    slab_energy=slab.get_potential_energy(),
    gas_energy=gas.get_potential_energy(),
)
