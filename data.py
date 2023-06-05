import numpy as np
import pandas as pd

df = pd.read_csv('emotions.csv')
print(df.head())
print(df.tail())

X = df.iloc[:, :-6].values
y = df.iloc[:, -6:].values

print(X)
print(y)