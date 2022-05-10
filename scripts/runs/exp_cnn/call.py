import pandas as pd
from run import main

data = pd.DataFrame("jobs.csv")
for row in data:
    main(**row.to_dict())
