import numpy as np
import math
import pandas as pd

def getStockDataVec(key):
	data = pd.read_csv("data/" + key + ".csv", sep=",", names=["Open", "High", "Low", "close", "Volume", "sin_days", "cos_days", "sin_time", "cos_time"])
	return data

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def getState(data, t, n, unrealized_gain, account_balance):
	# d = t - n + 1
	block = data[t:t + n].tolist() # if d >= 0 else np.add(-d * [data[0]], data[0:t + 1])
	#for i in range(len(block)):
       # block[i].append(unrealized_gain)
      #  block[i].append(account_balance)
	return np.array(block)