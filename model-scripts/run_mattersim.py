"""
run_mattersim.py — run in mlip-mattersim environment
conda activate mlip-mattersim
"""

import sys
import numpy as np
from ase.io import read
from mattersim.forcefield import MatterSimCalculator

# print("hello from mattersim!")  # check connection

slab_ads_frames = read(sys.argv[1], index=":")
slab = read(sys.argv[2])
gas = read(sys.argv[3])

calc = MatterSimCalculator()

for atoms in slab_ads_frames:
    atoms.calc = calc

slab.calc = calc
gas.calc = calc

energies_list = []
forces_list = []

for atoms in slab_ads_frames:
    energies_list.append(atoms.get_potential_energy())
    forces_list.append(atoms.get_forces())

np.savez(
    sys.argv[4],
    mlip_energies=np.array(energies_list),
    mlip_forces=np.array(forces_list),
    slab_energy=slab.get_potential_energy(),
    gas_energy=gas.get_potential_energy(),
)
