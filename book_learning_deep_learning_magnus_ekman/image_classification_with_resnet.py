import numpy as np
from keras.applications import resnet
from keras.preprocessing.image import image_utils
from keras.applications.resnet import decode_predictions
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import os

tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
	# load image and convert to tensor
	image = image_utils.load_img('./data/cat.jpg', target_size=(224, 224))
	image_np = image_utils.img_to_array(image)
	image_np = np.expand_dims(image_np, axis=0)

	# load pretrained model
	model = resnet.ResNet50(weights='imagenet')
	# standardize input data
	X = resnet.preprocess_input(image_np.copy())
	# do prediction
	y = model.predict(X)
	predicted_labels = decode_predictions(y)
	print(f'predictions: {predicted_labels}')

	# show image
	plt.imshow(np.uint8(image_np[0]))
	plt.show()
