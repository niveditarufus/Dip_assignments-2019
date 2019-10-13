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
im = cv2.imread('drops_input.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
lap_8 = np.array([[1,1,1],[1,-8,1],[1,1,1]])
a = sliding_window(im, lap_8)
cv2.imwrite('drops.jpg',a)