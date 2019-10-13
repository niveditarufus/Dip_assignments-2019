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

			a[x][y][0] = bilateral_filter(win_im[:,:,0])
			a[x][y][1] = bilateral_filter(win_im[:,:,1])
			a[x][y][2] = bilateral_filter(win_im[:,:,2])
	return a

def padding(im,kernel_row=3,kernel_col=3):
	image_row, image_col,ch = im.shape

	pad_height = int((kernel_row - 1) / 2)
	pad_width = int((kernel_col - 1) / 2)
	 
	padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width),ch))
	print(padded_image.shape)
	 
	padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = im
	return padded_image

def weighting(x,sigma=1):
	return math.exp(- (x ** 2) / (2 * sigma ** 2))

def bilateral_filter(win_im,sd=1,sr=1):
	w = 0 
	gk = 0
	i = int(win_im.shape[0]/2)
	# print(i)
	for k in range(win_im.shape[0]):
		for l in range(win_im.shape[1]):
			
			d = weighting((i-k),sd) * weighting((i-l),sd)
			# print(d)
			v = abs(win_im[i][i] - win_im[k][l])
			r = weighting(v,sr)
			gk = gk + (win_im[k][l]*(r*d))
			w = w + (r*d)

	return (gk/w)

im = cv2.imread('gt_sky.png')
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im = cv2.resize(im,(480,640))
padded_image = padding(im,3,3)
output = sliding_window(im,padded_image,3)
cv2.imwrite('gt _sky_ooo.jpg',output)
