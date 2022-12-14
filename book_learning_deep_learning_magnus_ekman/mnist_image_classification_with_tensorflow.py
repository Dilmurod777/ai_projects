import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import numpy as np
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(7)

EPOCHS = 20
BATCH_SIZE = 1

if __name__ == '__main__':
	# load training and test datasets
	mnist = keras.datasets.mnist
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	# standardize the data
	mean = np.mean(train_images)
	stddev = np.std(train_images)
	train_images = (train_images - mean) / stddev
	test_images = (test_images - mean) / stddev

	# one-hot encode labels
	train_labels = to_categorical(train_labels, num_classes=10)
	test_labels = to_categorical(test_labels, num_classes=10)

	# object used to initialize weights
	initializer = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=7)

	# create a Sequential model
	# 784 inputs
	# 2 Dense (fully connected) layers with 25 and 10 neurons
	# tanh as activation function for hidden layer
	# logistic (sigmoid) as activation function for output layer
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28, 28)),
		keras.layers.Dense(25, activation='tanh', kernel_initializer=initializer, bias_initializer='zeros'),
		keras.layers.Dense(10, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros')
	])

	# use stochastic gradient descent (SGD) with
	# learning rate of 0.01 and no other bells and whistles.
	# MSE as loss function and report accuracy during training
	opt = keras.optimizers.SGD(learning_rate=0.01)

	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

	# train the model for 20 epochs
	# shuffle (randomize) order
	# update weights after each example (batch_size=1)
	history = model.fit(train_images, train_labels,
											validation_data=(test_images, test_labels),
											epochs=EPOCHS,
											batch_size=BATCH_SIZE,
											verbose=2,
											shuffle=True)
