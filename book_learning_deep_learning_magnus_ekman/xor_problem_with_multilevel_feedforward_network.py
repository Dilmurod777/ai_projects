import numpy as np


def neuron_w(input_count):
	weights = np.zeros(input_count + 1)
	for i in range(1, input_count + 1):
		weights[i] = np.random.uniform(-1.0, 1.0)

	return weights


def show_learning():
	print("Current weights:")
	for i, w in enumerate(n_w):
		print(f'neuron {i}: w0={w[0]:5.2f}, w1={w[1]:5.2f}, w2={w[2]:5.2f}')
	print('-' * 30)


def forward_pass(x):
	global n_y

	n_y[0] = np.tanh(np.dot(n_w[0], x))  # neuron 0
	n_y[1] = np.tanh(np.dot(n_w[1], x))  # neuron 1

	n2_inputs = np.array([1.0, n_y[0], n_y[1]])  # 1.0 is bias
	z2 = np.dot(n_w[2], n2_inputs)
	n_y[2] = 1.0 / (1.0 + np.exp(-z2))


def backward_pass(y_truth):
	global n_error
	error_prime = -(y_truth - n_y[2])  # derivative of loss function
	derivative = n_y[2] * (1.0 - n_y[2])  # logistic derivative
	n_error[2] = error_prime * derivative
	derivative = 1.0 - n_y[0] ** 2  # tanh derivative
	n_error[0] = n_w[2][1] * n_error[2] * derivative
	derivative = 1.0 - n_y[1] ** 2  # tanh derivative
	n_error[1] = n_w[2][2] * n_error[2] * derivative


def adjust_weights(x):
	global n_w
	n_w[0] -= (x * LEARNING_RATE * n_error[0])
	n_w[1] -= (x * LEARNING_RATE * n_error[1])
	n2_inputs = np.array([1.0, n_y[0], n_y[1]])  # 1.0 is bias
	n_w[2] -= (n2_inputs * LEARNING_RATE * n_error[2])


if __name__ == "__main__":
	np.random.seed(3)
	LEARNING_RATE = 0.1
	index_list = [0, 1, 2, 3]  # used to randomize order

	# define training examples
	x_train = [np.array([1.0, -1.0, -1.0]),
						 np.array([1.0, -1.0, 1.0]),
						 np.array([1.0, 1.0, -1.0]),
						 np.array([1.0, 1.0, 1.0])]

	y_train = [0.0, 1.0, 1.0, 0.0]  # output (ground truth)
	n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]
	n_y = [0, 0, 0]
	n_error = [0, 0, 0]

	# network training loop
	all_correct = False
	while not all_correct:  # train until converged
		all_correct = True
		np.random.shuffle(index_list)  # randomize order
		for i in index_list:  # train on all examples
			forward_pass(x_train[i])
			backward_pass(y_train[i])
			adjust_weights(x_train[i])
			show_learning()  # show updated weights

		for i in range(len(x_train)):  # check if converged
			forward_pass(x_train[i])
			print(f"x1={x_train[i][1]:4.1f}, x2={x_train[i][2]:4.1f}, y={n_y[2]:.4f}")
			if (y_train[i] < 0.5 <= n_y[2]) or (y_train[i] >= 0.5 > n_y[2]):
				all_correct = False
