import sys
from ase.io import read

slab_ads_frames = read(sys.argv[1], index=":")
slab = read(sys.argv[2])
gas = read(sys.argv[3])

print("hello from mace!")