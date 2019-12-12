from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2 
import sys
import math

def BitQuantiseImage(image,bits=4):
	k= math.modf(256/(math.pow(2,bits)))[1]
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			image[x][y] = math.modf(k*math.modf(math.modf(image[x][y])[1]/k)[1])[1] 
			# image[x][y][1] = math.modf(k*math.modf(math.modf(image[x][y][1])[1]/k)[1])[1] 
			# image[x][y][2] = math.modf(k*math.modf(math.modf(image[x][y][2])[1]/k)[1])[1] 

	return image

im = cv2.imread('lena.jpg')# path needs to be channged all input images are available in the input folder
k_bits=int(sys.argv[1])
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img = BitQuantiseImage(im,k_bits)

cv2.imwrite("lena"+str(k_bits)+".jpg",im)
