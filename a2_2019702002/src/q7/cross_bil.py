import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

def sliding_window(image,padded_flash, padded_no_flash,window = 3):
	a = np.zeros(image.shape)
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			win_flash = padded_flash[x:x+window, y:y+window]
			win_noflash = padded_no_flash[x:x+window, y:y+window]
			a[x][y][0] = cross_bil_filter(win_flash[:,:,0],win_noflash[:,:,0])
			a[x][y][1] = cross_bil_filter(win_flash[:,:,1],win_noflash[:,:,1])
			a[x][y][2] = cross_bil_filter(win_flash[:,:,2],win_noflash[:,:,2])
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

def cross_bil_filter(win_flash, win_noflash,sd=2,sr=10):
	w = 0 
	gk = 0
	i = int(win_flash.shape[0]/2)
	for k in range(win_flash.shape[0]):
		for l in range(win_flash.shape[1]):
			
			d = weighting((i-k),sd) * weighting((i-l),sd) 
			v = abs(win_flash[i][i] - win_flash[k][l])
			r = weighting(v,sr) 
			print(r,d)
			gk = gk + ((r*d)* win_noflash[k][l])
			w = w + (r*d)
	return (gk/w)

flash = cv2.imread('pots_flash.jpg')
no_flash = cv2.imread("pots_no_flash.jpg")
window = 3
padded_flash = padding(flash,window,window)
padded_no_flash = padding(no_flash,window)
output = sliding_window(flash, padded_flash, padded_no_flash, window)
cv2.imwrite('proper.jpg',output)
