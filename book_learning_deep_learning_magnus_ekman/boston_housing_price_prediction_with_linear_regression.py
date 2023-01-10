import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import numpy as np
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 500
BATCH_SIZE = 16

# read and standardize the data
if __name__ == '__main__':
	boston_housing = keras.datasets.boston_housing
	(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()
	x_mean = np.mean(raw_x_train, axis=0)
	x_stddev = np.std(raw_x_train, axis=0)
	x_train = (raw_x_train - x_mean) / x_stddev
	x_test = (raw_x_test - x_mean) / x_stddev

	# create and train model
	model = Sequential()
	model.add(Dense(128,
									activation='relu',
									input_shape=[13],
									kernel_regularizer=l2(0.1),
									bias_regularizer=l2(0.1)))
	model.add(Dropout(0.2))
	model.add(Dense(128,
									activation='relu',
									kernel_regularizer=l2(0.1),
									bias_regularizer=l2(0.1)))
	model.add(Dropout(0.2))
	model.add(Dense(64,
									activation='relu',
									kernel_regularizer=l2(0.1),
									bias_regularizer=l2(0.1)))
	model.add(Dropout(0.2))
	model.add(Dense(1,
									activation='linear',
									kernel_regularizer=l2(0.1),
									bias_regularizer=l2(0.1)))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
	model.summary()

	history = model.fit(x_train, y_train,
											validation_data=(x_test, y_test),
											epochs=EPOCHS,
											batch_size=BATCH_SIZE,
											verbose=2,
											shuffle=True)

	# print first 4 predictions
	predictions = model.predict(x_test)
	for i in range(0, 4):
		print(f"Prediction: {predictions[i]}, true value: {y_test[i]}")
