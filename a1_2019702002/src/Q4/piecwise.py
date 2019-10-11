from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2 
import sys
import math

def piecewise_tranform(image):
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			k=(image[x][y])/255

			if k>=0.3 and k<=0.6:
				image[x][y] = (((k-0.3)*4/3) + 0.4)*255

			elif k>0.6 and k<=0.8:
				image[x][y] = (((k-0.8)*(-2)) + 0.4)*255
			else:
				image[x][y]=0
	return image
def step_tranform(image):
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			k = int((image[x][y]*10)/255)

			image[x][y] = 255*(k*0.2)
	return image

im = cv2.imread('lena.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im1 = im
im = piecewise_tranform(im)
img1 = step_tranform(im1)
cv2.imwrite("tranform2.jpg",im1)
