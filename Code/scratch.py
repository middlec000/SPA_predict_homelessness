import pandas as pd

data = {'one': [1,2,3], 'two':[2,3,4]}
df = pd.DataFrame(data=data)
df['three'] = df['one'] / (df['one'] + df['two'])

print(df)

print(3/7.0)