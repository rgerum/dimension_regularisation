import pandas as pd
from run import main

data = pd.read_csv("jobs.csv", index_col=0)
for i, row in data.iterrows():
    main(**row.to_dict())
