from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2 
import sys
import math

def negativeImage(image, maxintensity = 256):
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			image[x][y] = maxintensity - 1 - math.modf((image[x][y]))[1]
			# image[x][y][1] = maxintensity - 1 - int(image[x][y][1])
			# image[x][y][2] = maxintensity - 1 - int(image[x][y][2])

	return image

im = cv2.imread('lena7.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
max_int = np.amax(im)
img = negativeImage(im,max_int)
print(max_int)
cv2.imwrite("negative7.jpg",im)
