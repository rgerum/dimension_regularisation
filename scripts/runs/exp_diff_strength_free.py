import sys
import os
from pathlib import Path

for i in range(3):
    for strength in [0, 0.001, 0.01, 0.1, 1]:
        for sig in [1, -1]:
            reg = sig*strength
            os.system(f"python sbatch.py 1 a1_{i}_{reg} run_conv_augmentation3.py --reg1 {reg} --reg2 {0} --reg3 {0} --reg4 {0} --output diff_strength_free/alpha1_{i}_{reg}")
            os.system(f"python sbatch.py 1 a4_{i}_{reg} run_conv_augmentation3.py --reg1 {0} --reg2 {0} --reg3 {0} --reg4 {reg} --output diff_strength_free/alpha4_{i}_{reg}")
            os.system(f"python sbatch.py 1 aX_{i}_{reg} run_conv_augmentation3.py --reg1 {reg} --reg2 {reg} --reg3 {reg} --reg4 {reg} --output diff_strength_free/alphaX_{i}_{reg}")
            if reg == 0:
                break
