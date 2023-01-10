import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D
import numpy as np
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 128
BATCH_SIZE = 32

if __name__ == '__main__':
	# load dataset
	cifar_dataset = keras.datasets.cifar10
	(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

	# standardize dataset
	mean = np.mean(train_images)
	stddev = np.std(train_images)
	train_images = (train_images - mean) / stddev
	test_images = (test_images - mean) / stddev

	# change labels to one-hot
	train_labels = to_categorical(train_labels, num_classes=10)
	test_labels = to_categorical(test_labels, num_classes=10)

	# model with 2 convolutional and one fully connected layer
	model = Sequential()
	model.add(Conv2D(64, (4, 4),
									 activation='relu',
									 padding='same',
									 input_shape=(32, 32, 3)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (2, 2),
									 strides=(2, 2),
									 activation='relu',
									 padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3),
									 activation='relu',
									 padding='same'))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3),
									 activation='relu',
									 padding='same'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=2))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
								optimizer='adam',
								metrics=['accuracy'])
	model.summary()
	history = model.fit(train_images, train_labels,
											validation_data=(test_images, test_labels),
											epochs=EPOCHS,
											batch_size=BATCH_SIZE,
											verbose=2,
											shuffle=True)
