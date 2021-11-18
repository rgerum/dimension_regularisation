import sys
import os
from pathlib import Path

for i in range(3):
    for reg in [0, 0.001, 0.01, 0.1, 1]:
        os.system(f"python sbatch.py 1 gamma_{i}_{reg} run_conv_comp.py --reg1 {reg} --reg1value{reg} --reg_type=gamma --output comp/reg_gamma_{i}_{reg}")
        os.system(f"python sbatch.py 1 gamma_{i}_{reg} run_conv_comp.py --reg1 {reg} --reg1value{reg} --reg_type=fit --output comp/reg_fit_{i}_{reg}")
        if reg == 0:
            break