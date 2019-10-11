from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import cv2

def linContrastStretching(im, a=0, b=255,ahigh,alow):
	for x in range(im.shape[0]):
		for y in range(im.shape[1]):
			im[x][y] =  ((im[x][y] - alow)*(b-a)/(ahigh - alow))
	return im

grey_im = cv2.imread("lena.jpg")
grey_im = cv2.cvtColor(grey_im, cv2.COLOR_BGR2GRAY)
hi,wi = grey_im.shape
vis = np.zeros((hi, wi+wi))
vis[:hi, :wi] = grey_im


grey_im = linContrastStretching(grey_im,0, 255,230,35)
ho,wo = grey_im.shape

vis[:ho, wi:wi+wo] = grey_im
cv2.imwrite('output1.jpg',vis)
