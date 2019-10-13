from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

def sliding_window(image,padded_image,window = 3):
	a = np.zeros(image.shape)
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			win_im = padded_image[x:x+window, y:y+window]
			# print(image[x,y])
			
			a[x][y][0] = filter(win_im[:,:,0])
			a[x][y][1] = filter(win_im[:,:,1])
			a[x][y][2] = filter(win_im[:,:,2])
	print("median done")

	return a

def padding(im,kernel_row=3,kernel_col=3):
	image_row, image_col,ch = im.shape

	pad_height = int((kernel_row - 1) / 2)
	pad_width = int((kernel_col - 1) / 2)
	 
	padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width),ch))
	print(padded_image.shape)
	 
	padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = im
	return padded_image

def filter(win_im):
	x = np.sort(win_im.ravel())
	idx = x.shape[0]
	return x[int(idx/2)]

im = cv2.imread('panaroma.jpg')
padded_image = padding(im,3,3)
output = sliding_window(im,padded_image,3)
cv2.imwrite('panaroma_med.jpg',output)


