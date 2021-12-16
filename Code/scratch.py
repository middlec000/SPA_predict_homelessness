import pandas as pd
import numpy as np

data = {'id': [0,0,0,0,1,1,1,1,2,2,2], 'month':[0,1,2,3, 1,2,3,4, 1,2,3], 'enroll_date': [np.nan, np.nan, np.nan, np.nan, (2,3),(2,3),(2,3),(2,3), (1,2),(1,2),(1,2)]}
df = pd.DataFrame(data=data)

print(df)

df['min_enroll'] = df['enroll_date'].apply(lambda x: np.nan if x is np.nan else min(x))
df['drop_me'] = (df['month'] > df['min_enroll'])

print(df)

df = df[~df['drop_me']]
df = df.drop('drop_me', axis=1)

print(df)