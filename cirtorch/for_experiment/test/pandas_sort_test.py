import pandas as pd
import numpy as np


# a = np.random.choice(9, 9, replace=False).reshape((3, 3))
a = np.array([3,2,5,7,8,9,1,2,4]).reshape((3, 3))
df = pd.DataFrame(a)
print(df)
print(df.sort_values(0))
