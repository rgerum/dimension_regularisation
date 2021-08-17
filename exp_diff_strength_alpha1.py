import sys
import os
from pathlib import Path

for i in range(3):
    for strength in [0, 0.001, 0.01, 0.1, 1]:
        for sig in [1, -1]:
            reg = sig*strength
            os.system(f"python sbatch.py 1 a1_{i}_{reg} run_conv_augmentation2.py --reg1 {reg} --reg2 {0} --reg3 {0} --reg4 {0} --reg5 {0} --output diff_strength/alpha1_{i}_{reg}")
            os.system(f"python sbatch.py 1 a4_{i}_{reg} run_conv_augmentation2.py --reg1 {0} --reg2 {0} --reg3 {0} --reg4 {reg} --reg5 {0} --output diff_strength/alpha4_{i}_{reg}")
            if reg == 0:
                break
