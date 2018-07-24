import pandas as pd
import numpy as np 

data = pd.read_csv("data/AAPL(5year).csv")
time = np.zeros(len(data))
data = data.drop('Adj Close', axis=1)

data.insert(1, "Time", time)

dates = data.values[:,0]
data = data.drop('Date', axis=1)

for line in range(len(dates)):
    dates[line] = dates[line].replace('-','')

data.insert(0, "Date", dates)
print(data.head())

data.to_csv("data/AAPL(5year).csv", sep=",", index=False)