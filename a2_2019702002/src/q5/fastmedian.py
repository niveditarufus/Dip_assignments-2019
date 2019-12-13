from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import heapq


k = (9*9)/2

def sliding_window(image,padded_image,window = 3):
	a = np.zeros(image.shape)
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			win_im = padded_image[x:x+window, y:y+window]
			a[x][y] = kthSmallest(win_im.tolist())
	return a

def padding(im,kernel_row, kernel_col):
    image_row, image_col = im.shape
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = im
    return padded_image

def median(win_im):
	x = np.sort(win_im.ravel())
	idx = x.shape[0]
	return x[int(idx/2)]

def kthSmallest(input): 
    result = input[0]  
    heapq.heapify(list(result)) 
    for row in input[1:]: 
         for ele in row:
              heapq.heappush(result,ele) 
    kSmallest = heapq.nsmallest(k,result) 
    return kSmallest[-1]

im = cv2.imread('image.jpg')# path needs to be channged all input images are available in the input folder
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
padded_image = padding(im,9,9)
output = sliding_window(im,padded_image,9)
cv2.imwrite('median.jpg',output)# output images in the output folder can be used for refrence