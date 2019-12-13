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
im = cv2.imread('barbara.jpg')# path needs to be channged all input images are available in the input folder
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

mean = 127
var = 20
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(im.shape))
gauss = gauss.reshape(im.shape)
im = im + gauss
cv2.imwrite('noisy.jpg',im)# output images in the output folder can be used for refrence

sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
a = sliding_window(im, sobel_x)
cv2.imwrite('sobel_x_noise.jpg',a)# output images in the output folder can be used for refrence

sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
a = sliding_window(im, sobel_y)
cv2.imwrite('sobel_y_noise.jpg',a)# output images in the output folder can be used for refrence

prewit_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
a = sliding_window(im, prewit_x)
cv2.imwrite('prewit_x_noise.jpg',a)# output images in the output folder can be used for refrence

prewit_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
a = sliding_window(im, prewit_y)
cv2.imwrite('prewit_y_noise.jpg',a)# output images in the output folder can be used for refrence

robert_x = np.array([[0,-1],[1,0]])
a = sliding_window(im, robert_x,2)
cv2.imwrite('robert_x_noise.jpg',a)# output images in the output folder can be used for refrence

robert_y = np.array([[1,0],[0,-1]])
a = sliding_window(im, robert_y,2)
cv2.imwrite('robert_y_noise.jpg',a)# output images in the output folder can be used for refrence

lap_4 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
a = sliding_window(im, lap_4)
cv2.imwrite('lap_4_noise.jpg',a)# output images in the output folder can be used for refrence

lap_8 = np.array([[1,1,1],[1,-8,1],[1,1,1]])
a = sliding_window(im, lap_8)
cv2.imwrite('lap_8_noise.jpg',a)# output images in the output folder can be used for refrence
