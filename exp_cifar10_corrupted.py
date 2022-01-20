import sys
import os
from pathlib import Path

for i in range(3):
    for strength in [0, 0.001, 0.01, 0.1, 1]:
        reg = strength
        os.system(f"python sbatch.py 1 cifar10_{i}_{reg} run_cifar10_corrupted.py --reg1 {reg} --output exp_cifar10_corrupted/{i}_{reg}")
