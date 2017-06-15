import imutils
import sift
import cv2
import numpy as np
import time

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def slider_sift(im_name, w, b):
	image = cv2.imread(im_name)
	s = int(image.shape[0] / 2)
	(winW, winH) = (s, s)


	objects = []
	for (x, y, window) in sliding_window(image, stepSize=36,  windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		img = imutils.resize(window, width=50)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Computing keypoints and descriptors
		sift = cv2.SIFT()
		dense = cv2.FeatureDetector_create("Dense")
		kp = dense.detect(gray)
		kp,des = sift.compute(gray,kp)

		features = np.array(des).flatten()
		
		pred = np.dot(features, w.T) + b
		if pred > 3.2:
			objects.append([x, y, winW, winH])
			
	# Computing	centers of each object rectangle to trim redundant ones. The array is sorted and each rectangle
	# is compared with the next one. if they overlap, delete next rectangle

	i = 0
	"""while True:
		print "len(obj): ", len(objects)
		print "i: ", i
		if i >= len(objects) - 1:
			break
		center_xi = (2 * objects[i][0] + objects[i][2]) / 2
		center_xj = (2 * objects[i + 1][0] + objects[i + 1][2]) / 2
		center_yi = (2 * objects[i][1] + objects[i][3]) / 2
		center_yj = (2 * objects[i + 1][1] + objects[i + 1][3]) / 2

		if (abs(center_xi - center_xj) < 15 * objects[i][2]) and (abs(center_yi - center_yj) < 15 * objects[i][3]):
			del objects[i + 1]
		i += 1"""

	objects.reverse()
	clone = image.copy()
	for i in range(len(objects)):
		if i >= len(objects) - 1:
			break

		x = objects[i][0]
		y = objects[i][1]
		xx = objects[i][0] + objects[i][2]
		yy = objects[i][1] + objects[i][3]
		cv2.rectangle(clone, (x, y), (xx, yy), (0, 255, 0), 2)
		yield x, y, xx, yy

		center_xi = (2 * objects[i][0] + objects[i][2]) / 2
		center_xj = (2 * objects[i + 1][0] + objects[i + 1][2]) / 2
		center_yi = (2 * objects[i][1] + objects[i][3]) / 2
		center_yj = (2 * objects[i + 1][1] + objects[i + 1][3]) / 2
		if (abs(center_xi - center_xj) < 1 * objects[i][2]) and (abs(center_yi - center_yj) < 1 * objects[i][3]):
			break	

	cv2.imshow("Window", clone)
	cv2.waitKey(1)
	time.sleep(5)
