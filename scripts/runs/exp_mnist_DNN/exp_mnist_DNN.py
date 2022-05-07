import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

data = []
for i in range(3):
    for gamma in [False]:
        #for strength in [0, 0.001, 0.01, 0.1, 1]:
        for strength in [0, 0.1, 1]:
            #vs = [1.6, 1.8, 2.0, 2.2, 2.4]
            #if gamma is False:
            #vs = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
            #vs = [2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
            vs = np.arange(0.6, 4.1, 0.2)
            for reg1value in vs:
                for pca in np.arange(2, 11, 1):
                    reg = strength
                    data.append(dict(iter=i, gamma=gamma, class_count=pca, pca_dim=pca, reg_strength=strength, reg_target=reg1value, output=f"../../../results/expcifar5_DNN_extened_range_pca_classes/iter-{i}_gamma-{gamma}_pca{pca}_reg1-{strength}_reg1value-{reg1value}"))
data = pd.DataFrame(data)
print(data)
data.to_csv("jobs.csv")
