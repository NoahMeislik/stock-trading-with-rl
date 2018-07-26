import numpy as np
import math
import pandas as pd

def getStockDataVec(key):
	data = pd.read_csv("data/" + key + ".csv", sep=",", names=["Open", "High", "Low", "close", "Volume"])
	return data

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def getState(data, t, n):
	# d = t - n + 1
	block = data[t:t + n] # if d >= 0 else np.add(-d * [data[0]], data[0:t + 1])

	return block