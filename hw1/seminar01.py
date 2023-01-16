import pandas as pd
from sklearn.model_selection import KFold

kf = KFold(n_splits=3)
df = pd.read_csv('train.csv')

i = 1
for train, test in kf.split(df):
    df.iloc[train].to_csv(f'train_{i}.csv', index=False)
    df.iloc[test].to_csv(f'val_{i}.csv', index=False)
    i = i + 1
