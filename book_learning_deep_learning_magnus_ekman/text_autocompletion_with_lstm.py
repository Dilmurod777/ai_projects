import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
import logging
import os

tf.get_logger().setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

EPOCHS = 32
BATCH_SIZE = 256
INPUT_FILE_NAME = './data/frankenstein.txt'
WINDOW_LENGTH = 40
WINDOW_STEP = 3
BEAM_SIZE = 8
NUM_LETTERS = 11
MAX_LENGTH = 50

# open the input file
file = open(INPUT_FILE_NAME, 'r', encoding='utf-8')
text = file.read()
file.close()

# make lowercase and remove newline and extra spaces
text = text.lower()
text = text.replace('\n', ' ')
text = text.replace('  ', ' ')

# encode characters as indices
unique_chars = list(set(text))
char_to_index = dict((ch, index) for index, ch in enumerate(unique_chars))
index_to_char = dict((index, ch) for index, ch in enumerate(unique_chars))
encoding_width = len(char_to_index)

# create training examples
fragments = []
targets = []
for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
	fragments.append(text[i:i + WINDOW_LENGTH])
	targets.append(text[i + WINDOW_LENGTH])

# convert to one-hot encoded training data
X = np.zeros((len(fragments), WINDOW_LENGTH, encoding_width))
y = np.zeros((len(fragments), encoding_width))
for i, fragment in enumerate(fragments):
	for j, char in enumerate(fragment):
		X[i, j, char_to_index[char]] = 1
	target_char = targets[i]
	y[i, char_to_index[target_char]] = 1

# build and train model
model = Sequential()
model.add(LSTM(128,
							 return_sequences=True,
							 dropout=0.2,
							 recurrent_dropout=0.2,
							 input_shape=(None, encoding_width)))
model.add(LSTM(128,
							 dropout=0.2,
							 recurrent_dropout=0.2))
model.add(Dense(encoding_width, activation='softmax'))
model.compile(loss='categorical_crossentropy',
							optimizer='adam')
model.summary()
history = model.fit(X, y,
										validation_split=0.85,
										batch_size=BATCH_SIZE,
										epochs=EPOCHS,
										verbose=2,
										shuffle=True)

# create initial single eam represented by triple
# (probability, string, one-hot encoded string)
letters = 'the body '
one_hots = []
for i, char in enumerate(letters):
	x = np.zeros(encoding_width)
	x[char_to_index[char]] = 1
	one_hots.append(x)
beams = [(np.log(1.0), letters, one_hots)]

# predict NUM_LETTERS into the future
for i in range(NUM_LETTERS):
	minibatch_list = []
	# create minibatch from one-hot encodings, and predict
	for triple in beams:
		minibatch_list.append(triple[2])
	minibatch = np.array(minibatch_list)
	y_predict = model.predict(minibatch, verbose=0)
	new_beams = []
	for j, softmax_vec in enumerate(y_predict):
		triple = beams[j]
		# create BEAM_SIZE new beams from each existing beam
		for k in range(BEAM_SIZE):
			char_index = np.argmax(softmax_vec)
			new_prob = triple[0] + np.log(softmax_vec[char_index])
			new_letters = triple[1] + index_to_char[char_index]
			x = np.zeros(encoding_width)
			x[char_index] = 1
			new_one_hots = triple[2].copy()
			new_one_hots.append(x)
			new_beams.append((new_prob, new_letters, new_one_hots))
			softmax_vec[char_index] = 0

	# prune tree to only keam BEAM_SIZE most probable beams
	new_beams.sort(key=lambda tup: tup[0], reverse=True)
	beams = new_beams[0:BEAM_SIZE]

for item in beams:
	print(item[1])
