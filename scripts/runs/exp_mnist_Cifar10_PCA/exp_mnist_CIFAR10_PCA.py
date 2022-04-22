import sys
import os
from pathlib import Path

import pandas as pd

data = []
for i in range(3):
    for gamma in [True]:
        for strength in [1, 10]:
            #vs = [1.6, 1.8, 2.0, 2.2, 2.4]
            #if gamma is False:
            vs = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
            for reg1value in vs:
                for pca_dim in [2, 3, 5, 10]:
                    reg = strength
                    data.append(dict(iter=i, pca_dim=pca_dim, gamma=gamma, reg_strength=strength, reg_target=reg1value, output=f"../../../results/expcifar5_DNN_extened_range/iter-{i}_pca_dim-{pca_dim}_gamma-{gamma}_reg1-{strength}_reg1value-{reg1value}"))
data = pd.DataFrame(data)
print(data)
data.to_csv("jobs.csv")
