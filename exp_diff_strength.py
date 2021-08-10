import sys
import os
from pathlib import Path

for i in range(3):
    for strength in [0, 0.001, 0.01, 0.1, 1]:
        if strength == 0:
            reg = 0
            os.system(f"python sbatch.py 1 exp_diff_strength run_conv_augmentation2.py --reg1 {reg} --reg2 {reg} --reg3 {reg} --reg4{reg} --reg5{reg}")
        else:
            for sig in [1, -1]:
                reg = sig*strength
                os.system(f"python sbatch.py 1 exp_diff_strength run_conv_augmentation2.py --reg1 {reg} --reg2 {reg} --reg3 {reg} --reg4{reg} --reg5{reg}")
