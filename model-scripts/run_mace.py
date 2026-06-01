"""
run_mace.py — run in mlip-mace environment
conda activate mlip-mace
"""

import sys
import os
import numpy as np
from ase.io import read
from mace.calculators import mace_mp

print("hello from mace!")  # check connection

slab_ads_frames = read(sys.argv[1], index=":")
slab = read(sys.argv[2])
gas = read(sys.argv[3])

calc = mace_mp(model="medium", dispersion=False, default_dtype="float32")

for atoms in slab_ads_frames:
    atoms.calc = calc

slab.calc = calc
gas.calc = calc

energies_list = []
forces_list = []

for atoms in slab_ads_frames:
    energies_list.append(atoms.get_potential_energy())
    forces_list.append(atoms.get_forces())

print("saving to:", os.path.abspath(sys.argv[4]))
print("cwd:", os.getcwd())

np.savez(
    sys.argv[4],
    mlip_energies=np.array(energies_list),
    mlip_forces=np.array(forces_list),
    slab_energy=slab.get_potential_energy(),
    gas_energy=gas.get_potential_energy(),
)
