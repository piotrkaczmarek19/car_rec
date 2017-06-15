from PIL import Image
import os
import numpy as np 
import matplotlib.pylab as plt
import cv2
import pickle

def process_image(imagename, resultname):
	img = cv2.imread(imagename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Computing keypoints and descriptors
	sift = cv2.SIFT()
	kp = sift.detect(gray,None)
	kp,des = sift.compute(gray,kp)

	# writing sift array
	sift_array =[]
	for i in range(len(kp)):
		line = [[kp[i].pt[0], kp[i].pt[1], kp[i].size, kp[i].angle] + list(des[i])]
		sift_array += line
	print len(sift_array)
	np.savetxt(resultname, sift_array)
	print 'processed', imagename, 'to', resultname

def read_features_from_file(filename):
	# Read features properties and return in matrix form
	f = np.loadtxt(filename)

	return f[:, :4], f[:, 4:] # return feature locations, descriptors

def write_features_to_file(filenam, locs, desc):
	np.savetxt(filename, np.hstack((locs, desc)))

def plot_features(im, locs, circle=False):
	# Show image with features
	# Input: image as array, locs (row, col, scale, orientation of each feature)

	def draw_circle(c,r):
		t = plt.arange(0, 1.01, .01) * 2 * np.pi 
		x = r/8 * np.cos(t) + c[0]
		y = r/8 * np.sin(t) + c[1]

		plt.plot(x, y, 'b', linewidth=2)

	plt.imshow(im)
	if circle:
		for p in locs:
			draw_circle(p[:2], p[2])
	else:
		plt.plot(locs[:,0], locs[:,1], 'ob')
	plt.axis('off')

def match(desc1, desc2):
	# For each descriptor in the first image, 
	# select its match in the second image 

	desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
	desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

	dist_ratio = 0.6
	desc1_size = desc1.shape

	matchscores = np.zeros((desc1_size[0], 1), 'int')
	desc2t = desc2.T # precompute matrix transpose
	for i in range(desc1_size[0]):
		dotprods = np.dot(desc1[i,:], desc2t)
		dotprods = 0.9999 * dotprods

		# inverse cosine and sort, return index for features in second image
		indx = np.argsort(np.arccos(dotprods))

		# check if nearest neighbor has angle less than dist_ratio times 2nd
		if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
			matchscores[i] = int(indx[0])
	return matchscores

def match_twosided(desc1, desc2):

	matches_12 = match(desc1, desc2)
	matches_21 = match(desc2, desc1)

	ndx_12 = matches_12.nonzero()[0]

	# Remove matches that are not symmetric
	for n in ndx_12:
		if matches_21[int(matches_12[n])] != n:
			matches_12[n] = 0
	return matches_12

def appendimages(im1, im2):
	# Return a new image that appends the two images side by side

	# select the image with the fewest rows and fill in enough empty rows
	rows1 = im1.shape[0]
	rows2 = im2.shape[0]

	# If one image has more rows than the other, add missing rows to smaller one
	if rows1 < rows2:
		im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
	elif rows2 > rows1:
		im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im1.shape[1]))), axis=0)

	return np.concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
	# Show a figure with lines joining the accepted matches 
	# Input: im1, im2 as numpy arrays, loc1, loc2 (feature locations), matchscores (output from match()),
	# show_below (if images should be show alongside matches)

	im3 = appendimages(im1, im2)
	if show_below:
		im3 = np.vstack((im3, im3))
	
	plt.imshow(im3, cmap='Greys_r')

	cols1 = im1.shape[1]
	# Analysing correlation scores found in match() and plotting lines
	for i, m in enumerate(matchscores):
		if m > 0:			
			l1 = locs1[i].flatten()
			l2 = locs2[m].flatten()
			
			plt.plot([l1[0], l2[0] + cols1], [l1[1], l2[1]], 'c')
			plt.axis('off')