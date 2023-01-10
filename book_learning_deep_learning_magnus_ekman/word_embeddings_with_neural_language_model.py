import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
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
PREDICT_LENGTH = 3
MAX_WORDS = 10000
EMBEDDING_WIDTH = 100

# open and read file
file = open(INPUT_FILE_NAME, 'r', encoding='utf-8')
text = file.read()
file.close()

# make lowercase and split into individual words
text = text_to_word_sequence(text)

# create training examples
fragments = []
targets = []
for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
	fragments.append(text[i: i + WINDOW_LENGTH])
	targets.append(text[i + WINDOW_LENGTH])

# convert indices
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='UNK')
tokenizer.fit_on_texts(text)
fragments_indexed = tokenizer.texts_to_sequences(fragments)
targets_indexed = tokenizer.texts_to_sequences(targets)

# convert to appropriate input and output formats
X = np.array(fragments_indexed, dtype=np.int_)
y = np.zeros((len(targets_indexed), MAX_WORDS))
for i, target_index in enumerate(targets_indexed):
	y[i, target_index] = 1

# build and train model
training_model = Sequential()
training_model.add(Embedding(
	output_dim=EMBEDDING_WIDTH,
	input_dim=MAX_WORDS,
	mask_zero=True,
	input_length=None
))
training_model.add(LSTM(128,
												return_sequences=True,
												dropout=0.2,
												recurrent_dropout=0.2))
training_model.add(LSTM(128,
												dropout=0.2,
												recurrent_dropout=0.2))
training_model.add(Dense(128, activation='relu'))
training_model.add(Dense(MAX_WORDS, activation='softmax'))
training_model.compile(loss='categorical_crossentropy',
											 optimizer='adam')
training_model.summary()
history = training_model.fit(X, y,
														 validation_split=0.05,
														 batch_size=BATCH_SIZE,
														 epochs=EPOCHS,
														 verbose=2,
														 shuffle=True)

# build stateful model used for prediction
inference_model = Sequential()
inference_model.add(Embedding(
	output_dim=EMBEDDING_WIDTH,
	input_dim=MAX_WORDS,
	mask_zero=True,
	batch_input_shape=(1, 1)
))
inference_model.add(LSTM(128,
												 return_sequences=True,
												 dropout=0.2,
												 recurrent_dropout=0.2,
												 stateful=True))
inference_model.add(LSTM(128,
												 dropout=0.2,
												 recurrent_dropout=0.2,
												 stateful=True))
inference_model.add(Dense(128, activation='relu'))
inference_model.add(Dense(MAX_WORDS, activation='softmax'))
weights = training_model.get_weights()
inference_model.set_weights(weights)

# provide beginning of sentence and
# predict next words in a greedy manner
first_words = ['i', 'saw']
first_words_indexed = tokenizer.texts_to_sequences(first_words)
inference_model.reset_states()
predicted_string = ''

# feed initial words
for i, word_index in enumerate(first_words_indexed):
	x = np.zeros((1, 1), dtype=np.int_)
	x[0][0] = word_index[0]
	predicted_string += first_words[i]
	predicted_string += ' '
	y_predict = inference_model.predict(x, verbose=0)[0]

# predict PREDICT_LENGTH words
for i in range(PREDICT_LENGTH):
	new_word_index = np.argmax(y_predict)
	word = tokenizer.sequences_to_texts([[new_word_index]])
	x[0][0] = new_word_index
	predicted_string += word[0]
	predicted_string += ' '
	y_predict = inference_model.predict(x, verbose=0)[0]

print("Predicted string: ", predicted_string)

# explore embedding similarities
embeddings = training_model.layers[0].get_weights()[0]
lookup_words = ['the', 'saw', 'see', 'of', 'and', 'monster', 'frankenstein', 'read', 'eat']

for lookup_word in lookup_words:
	lookup_word_indexed = tokenizer.texts_to_sequences([lookup_word])
	print('words close to: ', lookup_word)
	lookup_embedding = embeddings[lookup_word_indexed[0]]
	word_indices = {}

	# calculate distances
	for i, embedding in enumerate(embeddings):
		distance = np.linalg.norm(embedding - lookup_embedding)
		word_indices[distance] = i

	# print sorted by distance
	for distance in sorted(word_indices.keys())[:5]:
		word_index = word_indices[distance]
		word = tokenizer.sequences_to_texts([[word_index]])[0]
		print(word + ': ', distance)
	print('')
