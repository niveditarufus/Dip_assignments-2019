from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

def sliding_window(image, mask,window = 3):
	a = np.zeros(image.shape)
	for x in range(1, image.shape[0]-1):
		for y in range(1, image.shape[1]-1):
			win_im = image[x-1:x+window-1, y-1:y+window-1]
			win_im = np.multiply(win_im, mask)
			a[x][y] = np.sum(win_im)
			
	return a
im = cv2.imread('barbara.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
a = sliding_window(im, sobel_x)
cv2.imwrite('sobel_x.jpg',a)

sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
a = sliding_window(im, sobel_y)
cv2.imwrite('sobel_y.jpg',a)

prewit_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
a = sliding_window(im, prewit_x)
cv2.imwrite('prewit_x.jpg',a)

prewit_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
a = sliding_window(im, prewit_y)
cv2.imwrite('prewit_y.jpg',a)

robert_x = np.array([[0,-1],[1,0]])
a = sliding_window(im, robert_x,2)
cv2.imwrite('robert_x.jpg',a)

robert_y = np.array([[1,0],[0,-1]])
a = sliding_window(im, robert_y,2)
cv2.imwrite('robert_y.jpg',a)

lap_4 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
a = sliding_window(im, lap_4)
cv2.imwrite('lap_4.jpg',a)

lap_8 = np.array([[1,1,1],[1,-8,1],[1,1,1]])
a = sliding_window(im, lap_8)
cv2.imwrite('lap_8.jpg',a)
