#!/usr/bin/env python
from __future__ import division
import numpy as np 
from PIL import Image
import os
import cv2
import dsift, sift, sliding_window
from numpy import linalg
import math
import pickle
import matplotlib.pylab as plt

# cleaning up after training
def deleteFeatFile(folder, format):
	files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(format)]
	for f in files:
		os.remove(f)

def getFeatures(folder, filename):
	labels = [(1 if ndx < len(os.listdir(folder)) else -1) for ndx, f in enumerate(im_names)] # first half are pos, second neg
	
	# writing and listing individual feature files
	
	featfile = filename[:-3]+'dsift'
	dsift.process_image_dsift(filename, featfile, 10, 10, resize=(50,50))
	featlist = loadDataset(folder, '.dsift')

	os.chmod(featfile, 0o777)
	os.chmod("tmp.frame", 0o777)
	os.chmod("tmp.pgm", 0o777)

	# gathering feature data into numpy array
	features = []
	counter = 0
	for featfile in featlist:
		
		counter += 1
		l, d = sift.read_features_from_file(featfile)

		features.append(d.flatten())
		#print "Processed: ", featfile, "image ", counter, " out of ", len(im_names)

	features = np.array(features)
	return features, labels
dir_path = os.path.dirname(os.path.realpath(__file__))


def loadDataset(folder, format=""):
	im_array = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(format)]
	return im_array

im_names = loadDataset(dir_path, 'jpg') 

deleteFeatFile(dir_path, 'dsift')
deleteFeatFile(dir_path, 'frame')
deleteFeatFile(dir_path, 'pgm')

with open("coeff.pickle", "rb") as f:
	w, b = pickle.load(f)

for image in im_names:
	for key, coord in enumerate(sliding_window.slider_sift(image, w, b)):
		print coord



deleteFeatFile(dir_path, 'dsift')
deleteFeatFile(dir_path, 'frame')
deleteFeatFile(dir_path, 'pgm')

