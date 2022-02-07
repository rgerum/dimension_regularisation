import sys
import os
from pathlib import Path

for i in range(3):
    for strength in [0, 0.001, 0.01, 0.1, 1]:
        for sig in [1, -1]:
            reg = sig*strength
            os.system(f"python sbatch.py 1 exp_diff_strength_{i}_{reg} run_conv_augmentation2.py --reg1 {reg} --reg2 {reg} --reg3 {reg} --reg4 {reg} --reg5 {reg} --output diff_strength/{i}_{reg}")
            if reg == 0:
                break
