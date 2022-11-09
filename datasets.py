import csv
from operator import delitem
from matplotlib.pyplot import table
import tensorflow as tf
import numpy as np

filenames = ['data/glucose (1).csv', 'data/glucose (2).csv', 'data/glucose (3).csv', 'data/glucose (4).csv', 'data/glucose (5).csv', 'data/glucose (6).csv', 'data/glucose (7).csv', 'data/glucose (8).csv']

def load_glucose_data():
	values: list[float] = []
	for filename in filenames:
		A = np.genfromtxt(filename, dtype=int, delimiter=',').astype(np.float32)
		values.extend(A[:,1])
	samples = len(values) // 16
	values = np.array(values[:samples * 16])
	values = np.reshape(values, (samples, 16))
	values = values[values.min(axis=1)>=0,:]
	#values = (values - min(values)) / (max(values) - min(values))
	values /= values.sum(axis=1)[:,np.newaxis]
	return values