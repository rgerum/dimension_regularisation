import sys
import os
from pathlib import Path

import pandas as pd

data = []
for i in range(3):
    for gamma in [True]:
        for strength in [0, 0.001, 0.01, 0.1, 1]:
            for reg1value in [0.6, 0.8, 1.0, 1.2, 1.4]:
                reg = strength
                data.append(dict(iter=i, gamma=gamma, reg1=strength, reg1value=reg1value, output=f"../../../results/expcifar5_DNN/iter-{i}_gamma-{gamma}_reg1-{strength}_reg1value-{reg1value}"))
data = pd.DataFrame(data)
print(data)
data.to_csv("jobs.csv")
