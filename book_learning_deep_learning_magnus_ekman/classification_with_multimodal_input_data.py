import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical, pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Input, Embedding, LSTM, Flatten, Concatenate, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


# function to create second modality
def create_text(tokenizer, labels):
	text = []
	for i, label in enumerate(labels):
		if i % 2 == 0:
			if label < 5:
				text.append('lower half')
			else:
				text.append('upper half')
		else:
			if label % 2 == 0:
				text.append('even number')
			else:
				text.append('odd number')

	text = tokenizer.texts_to_sequences(text)
	text = pad_sequences(text)
	return text


# create second modality from training and test set
vocabulary = ['lower', 'upper', 'half', 'even', 'odd', 'number']
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(vocabulary)
train_text = create_text(tokenizer, train_labels)
test_text = create_text(tokenizer, test_labels)

# create model with functional API
image_input = Input(shape=(28, 28))
text_input = Input(shape=(2,))

embedding_layer = Embedding(output_dim=EMBEDDING_WIDTH,
														input_dim=MAX_WORDS)
lstm_layer = LSTM(8)
flatten_layer = Flatten()
concat_layer = Concatenate()
dense_layer = Dense(25, activation='relu')
output_layer = Dense(10, activation='softmax')

# connect layers
embedding_output = embedding_layer(text_input)
lstm_output = lstm_layer(embedding_output)
flatten_output = flatten_layer(image_input)
concat_output = concat_layer([lstm_output, flatten_output])
dense_output = dense_layer(concat_output)
outputs = output_layer(dense_output)

# build and train model
model = Model([image_input, text_input], outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit([train_images, train_text], train_labels,
										validation_data=([test_images, test_text], test_labels),
										epochs=EPOCHS, batch_size=64, verbose=2, shuffle=True)

# print input modalities and output for one test example
print(test_labels[0])
print(tokenizer.sequences_to_texts([test_text[0]]))
plt.figure(figsize=(1, 1))
plt.imshow(test_images[0], cmap=plt.get_cmap('gray'))
plt.show()

# predict test example
y = model.predict([test_images[0:1], np.array(tokenizer.texts_to_sequences(['upper half']))])[0]
print('Predictions with correct input:')
for i in range(len(y)):
	index = y.argmax()
	print(f'Digit: {index}, probability: {y[index]:5.2e}')
	y[index] = 0

# predict same test example but with modified textual description
print('\n Predictions with incorrect input:')
y = model.predict([test_images[0:1], np.array(tokenizer.texts_to_sequences(['lower half']))])[0]
for i in range(len(y)):
	index = y.argmax()
	print(f'Digit: {index}, probability: {y[index]:5.2e}')
	y[index] = 0
