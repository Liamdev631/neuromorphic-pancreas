from tensorflow import keras
import numpy as np
import tensorflow as tf

maskval = np.zeros(16, dtype=np.float32)
maskval[0:-2] = 1
print('mask', maskval)
maskval = tf.convert_to_tensor(maskval)
def inputMask(x):
	return x * maskval

def make_dense_autoencoder():
	return keras.models.Sequential([
		keras.Input(shape=(16,), name='glucose'),
		keras.layers.Lambda(inputMask),
		#keras.layers.Dropout(2/16, name='dropout'),
		keras.layers.Dense(12, activation='relu', name='encoder1'),
		keras.layers.Dense(4, activation='relu', name='encoder2'),
		keras.layers.Dense(12, activation='relu', name='decoder1'),
		keras.layers.Dense(16, activation='relu', name='decoder2')
	])

def make_conv_autoencoder():
	return keras.models.Sequential([
		keras.Input(shape=(16,), name='glucose'),
		keras.layers.Lambda(inputMask),
		keras.layers.Dense(4, activation='sigmoid', name='encoder'),
		keras.layers.Dense(16, activation='sigmoid', name='decoder')
	])
