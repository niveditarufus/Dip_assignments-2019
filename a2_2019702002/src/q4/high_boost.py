from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

def sliding_window(image,padded_image, mask,window = 3):
	a = np.zeros(image.shape)
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			win_im = padded_image[x:x+window, y:y+window]
			win_im = np.multiply(win_im, mask)
			a[x][y] = np.sum(win_im)
	return a

def padding(im,mask):
	image_row, image_col = im.shape
	kernel_row, kernel_col = mask.shape
	 
	pad_height = int((kernel_row - 1) / 2)
	pad_width = int((kernel_col - 1) / 2)
	 
	padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
	 
	padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = im
	return padded_image


def generate_mask(weight,size = 3):
	mask = -1*(np.ones([size,size]))
	x = int(size/2)
	mask[x][x] = (size**2*weight) - 1
	print(mask)
	return mask

im = cv2.imread('ice.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_ice.jpg",im)
mask = generate_mask (1.1,3)
padded_image = padding(im,mask)
output = sliding_window(im,padded_image,mask,1)
cv2.imwrite('trial_ice3.jpg',output)

 
