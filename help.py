import pandas as pd
import numpy as np

data=pd.read_csv('dataset.csv')
data2=data.append(data)
data3=data2.append(data2)
data3.to_csv('file1.csv')

df=pd.read_csv('file1.csv')
df2=df.append(df)
df2.to_csv('data.csv')

df3=pd.read_csv('data.csv')