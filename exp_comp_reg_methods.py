import sys
import os
from pathlib import Path

for i in range(3):
    for reg in [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
        os.system(f"python sbatch.py 1 gamma_{i}_{reg} run_conv_comp.py --reg1 {reg} --reg2 {0} --reg3 {0} --reg4 {0} --reg_type gamma --output comp2b/reg_gamma_{i}_{reg}")
        os.system(f"python sbatch.py 1 fit_{i}_{reg} run_conv_comp.py --reg1 {reg} --reg2 {0} --reg3 {0} --reg4 {0} --reg_type fit --output comp2b/reg_fit_{i}_{reg}")

        os.system(f"python sbatch.py 1 gamma_all_{i}_{reg} run_conv_comp.py --reg1 {reg} --reg2 {reg} --reg3 {reg} --reg4 {reg} --reg_type gamma --output comp2b/reg_all_gamma_{i}_{reg}")
        os.system(f"python sbatch.py 1 fit_all_{i}_{reg} run_conv_comp.py --reg1 {reg} --reg2 {reg} --reg3 {reg} --reg4 {reg} --reg_type fit --output comp2b/reg_all_fit_{i}_{reg}")
