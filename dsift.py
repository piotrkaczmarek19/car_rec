from PIL import Image
import os
import numpy as np 
import sift
import cv2

def process_image_dsift(imagename, resultname, size=20, steps=10, force_orientation=False, resize=None):
	# Process an image with densely sampled SIFT descriptors and save the results in a file. Optional
	# input: size of features, steps between locations, forcing computation of descriptors orientation 
	# (False means all are oriented upwards), tuple for resizing the image 

	im = Image.open(imagename).convert('L')
	if resize != None:
		im = im.resize(resize)
	m, n = im.size

	if imagename[-3:] != 'pgm':
		# create a pgm file
		im.save('tmp.pgm')
		imagename = 'tmp.pgm'

	# create frames and save to temporary file
	scale = size / 3.0
	x, y = np.meshgrid(range(steps, m, steps), range(steps, n, steps))
	xx, yy = x.flatten(), y.flatten()
	frame = np.array([xx, yy, scale * np.ones(xx.shape[0]), np.zeros(xx.shape[0])])
	np.savetxt('tmp.frame', frame.T, fmt="%03.3f")

	

	img = cv2.imread(imagename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Computing keypoints and descriptors
	sift = cv2.SIFT()
	dense = cv2.FeatureDetector_create("Dense")
	kp = dense.detect(gray)
	kp,des = sift.compute(gray,kp)

	# writing sift array
	sift_array =[]
	for i in range(len(kp)):
		line = [[kp[i].pt[0], kp[i].pt[1], kp[i].size, kp[i].angle] + list(des[i])]
		sift_array += line
	np.savetxt(resultname, sift_array)


