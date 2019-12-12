from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2 
import sys
import math

def get_histogram(image,i):
	plt.subplot(2,1,i)
	count, bins, patches = plt.hist(image.ravel(), bins=256)
	return count

def equalise(image,a):
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			image[x][y] = a[image[x][y]]
	return image



im = cv2.imread('church.png')# path needs to be channged all input images are available in the input folder

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite("test_church_gray.jpg",im)# output images in the output folder can be used for refrence

count = get_histogram(im,1)
print(len(count))
a = count/im.size
a = np.cumsum(a)
a = a*255
im = equalise(im,a)
get_histogram(im,2)
cv2.imwrite("test_church.jpg",im)# output images in the output folder can be used for refrence
plt.show()
