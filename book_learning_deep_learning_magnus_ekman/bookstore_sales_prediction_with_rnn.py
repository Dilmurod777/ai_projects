import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import logging
import os

tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 100
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.8
MIN = 12
FILE_NAME = './data/bookstore_sales/bookstore_sales.csv'


def readFile(file_name):
	file = open(file_name, 'r', encoding='utf-8')
	next(file)
	data = []

	for line in file:
		values = line.split(',')
		data.append(float(values[1]))

	file.close()
	return np.array(data, dtype=np.float32)


# read data and split into training and test data
sales = readFile(FILE_NAME)
months = len(sales)
split = int(months * TRAIN_TEST_SPLIT)
train_sales = sales[:split]
test_sales = sales[split:]

# plot dataset
x = range(len(sales))
plt.plot(x, sales, 'r-', label='book sales')
plt.title('Book store sales')
plt.axis([0, 339, 0.0, 3000.0])
plt.xlabel('Months')
plt.ylabel('Sales (millions $)')
plt.legend()
plt.show()

# plot naive prediction
test_output = test_sales[MIN:]
naive_prediction = test_sales[MIN - 1:-1]
x = range(len(test_output))
plt.plot(x, test_output, 'g-', label='test_output')
plt.plot(x, naive_prediction, 'm-', label='naive prediction')
plt.title('Book store sales')
plt.axis([0, len(test_output), 0.0, 3000.0])
plt.xlabel('months')
plt.ylabel('Monthly book store sales')
plt.legend()
plt.show()

# standardize the data
# use only training seasons to compute mean and stddev
mean = np.mean(train_sales)
stddev = np.std(train_sales)
train_sales_std = (train_sales - mean) / stddev
test_sales_std = (test_sales - mean) / stddev

# create training examples
train_months = len(train_sales)
train_X = np.zeros((train_months - MIN, train_months - 1, 1))
train_Y = np.zeros((train_months - MIN, 1))
for i in range(train_months - MIN):
	train_X[i, -(i + MIN):, 0] = train_sales_std[0:i + MIN]
	train_Y[i, 0] = train_sales_std[i + MIN]

# create test examples
test_months = len(test_sales)
test_X = np.zeros((test_months - MIN, test_months - 1, 1))
test_Y = np.zeros((test_months - MIN, 1))
for i in range(test_months - MIN):
	test_X[i, -(i + MIN):, 0] = test_sales_std[0:i + MIN]
	test_Y[i, 0] = test_sales_std[i + MIN]

# create RNN model
model = Sequential()
model.add(SimpleRNN(129,
										activation='relu',
										input_shape=(None, 1)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',
							optimizer='adam',
							metrics=['mean_absolute_error'])
model.summary()
history = model.fit(train_X, train_Y,
										epochs=EPOCHS,
										batch_size=BATCH_SIZE,
										verbose=2,
										shuffle=True)

# create naive prediction based on standardized data
test_output = test_sales_std[MIN:]
naive_prediction = test_sales_std[MIN - 1:-1]
mean_squared_error = np.mean(np.square(naive_prediction - test_output))
mean_abs_error = np.mean(np.abs(naive_prediction - test_output))
print(f'naive test mse: {mean_squared_error}')
print(f'naive test mean abs: {mean_abs_error}')

# use trained model to predict the test data
predicted_test = model.predict(test_X, len(test_X))
predicted_test = np.reshape(predicted_test, (len(predicted_test)))
predicted_test = predicted_test * stddev + mean

# plot test prediction
x = range(len(test_sales) - MIN)
plt.plot(x, predicted_test, 'm-', label='predicted test_output')
plt.plot(x, test_sales[-(len(test_sales) - MIN):], 'g-', label='actual test_output')
plt.title('Book sales')
plt.axis([0, 55, 0.0, 3000.0])
plt.xlabel('months')
plt.ylabel('Predicted book sales')
plt.legend()
plt.show()
