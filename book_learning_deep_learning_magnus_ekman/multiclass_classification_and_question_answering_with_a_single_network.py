import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical, pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Input, Embedding, LSTM, Flatten, Concatenate, Dense
from keras.models import Model
import numpy as np
import logging
import os

tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LEVEL_LOG'] = '2'

EPOCHS = 20
MAX_WORDS = 8
EMBEDDING_WIDTH = 4

# load training and test datasets
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# standardize the data
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev


# function to create and answer text
def create_question_answer(tokenizer, labels):
	text = []
	answers = np.zeros(len(labels))
	for i, label in enumerate(labels):
		question_num = i % 4
		if question_num == 0:
			text.append('lower half')
			if label < 5:
				answers[i] = 1.0
		elif question_num == 1:
			text.append('upper half')
			if label >= 5:
				answers[i] = 1.0
		elif question_num == 2:
			text.append('even number')
			if label % 2 == 0:
				answers[i] = 1.0
		elif question_num == 3:
			text.append('odd number')
			if label % 2 == 1:
				answers[i] = 1.0
	text = tokenizer.texts_to_sequences(text)
	text = pad_sequences(text)
	return text, answers


# create second modality for training and test set
vocabulary = ['lower', 'upper', 'half', 'even', 'odd', 'number']
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(vocabulary)
train_text, train_answers = create_question_answer(tokenizer, train_labels)
test_text, test_answers = create_question_answer(tokenizer, test_labels)

# create model with functional API
image_input = Input(shape=(28, 28))
text_input = Input(shape=(2,))

# declare layers
embedding_layer = Embedding(output_dim=EMBEDDING_WIDTH,
														input_dim=MAX_WORDS)
lstm_layer = LSTM(8)
flatten_layer = Flatten()
concat_layer = Concatenate()
dense_layer = Dense(25, activation='relu')
class_output_layer = Dense(10, activation='softmax')
answer_output_layer = Dense(1, activation='sigmoid')

# connect layers
embedding_output = embedding_layer(text_input)
lstm_output = lstm_layer(embedding_output)
flatten_output = flatten_layer(image_input)
concat_output = concat_layer([lstm_output, flatten_output])
dense_output = dense_layer(concat_output)
class_outputs = class_output_layer(dense_output)
answer_outputs = answer_output_layer(dense_output)

# build and train model
model = Model([image_input, text_input],
							[class_outputs, answer_outputs])
model.compile(loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
							optimizer='adam', metrics=['accuracy'], loss_weights=[0.5, 0.5])
model.summary()
history = model.fit([train_images, train_text],
										[train_labels, train_answers],
										validation_data=([test_images, test_text], [test_labels, test_answers]),
										epochs=EPOCHS, batch_size=64, verbose=2, shuffle=True)
