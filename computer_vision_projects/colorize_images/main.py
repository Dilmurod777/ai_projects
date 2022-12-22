import os

import numpy as np
import cv2

prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'

bw_images_path = './files/black_white/'
c_images_path = './files/colorized/'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

for file_path in os.listdir(bw_images_path):
	if os.path.isfile(bw_images_path + file_path):
		bw_image = cv2.imread(bw_images_path + file_path)
		normalized = bw_image.astype('float32') / 255.0
		lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2Lab)
		resized = cv2.resize(lab, (224, 224))
		L = resized[:, :, 0]
		L -= 50
		
		net.setInput(cv2.dnn.blobFromImage(L))
		ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
		
		ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
		L = cv2.split(lab)[0]
		
		colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
		colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
		
		colorized = (255.0 * colorized).astype("uint8")
		
		cv2.imwrite(c_images_path + file_path, colorized)
		print("Finished: " + file_path)
