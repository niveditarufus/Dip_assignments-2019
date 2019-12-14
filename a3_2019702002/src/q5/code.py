import numpy as np
import cv2
import math
import cmath
import sys
np.set_printoptions(threshold=sys.maxsize)

def filter(img):
	H = np.array([[0,1,0],[1,2,1],[0,1,0]])
	sz = (img.shape[0] - H.shape[0], img.shape[1] -H.shape[1])  # total 
	H= np.pad(H, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)),'constant')
	return H

def padding(im,kernel_row = 3, kernel_col = 3):
	image_row, image_col = im.shape
	pad_height = int((kernel_row - 1) / 2)
	pad_width = int((kernel_col - 1) / 2)
	padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
	padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = im
	return padded_image

im = cv2.imread('lena.jpg')# path needs to be channged all input images are available in the input folder
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = padding(im)


im_fft = np.fft.fft2(im)
im_fft = np.fft.fftshift(im_fft)
cv2.imwrite("input.jpg",np.fft.ifft2(np.fft.fftshift(im_fft)).astype("uint8"))# output images in the output folder can be used for refrence


H = filter(im_fft)
cv2.imwrite("filter.jpg",10*np.log(abs(H)+1))# output images in the output folder can be used for refrence
o = np.real(np.fft.fftshift(np.fft.ifft2(im_fft * np.fft.fftshift(np.fft.fft2(H)))))+np.imag(np.fft.fftshift(np.fft.ifft2((im_fft) * np.fft.fftshift(np.fft.fft2(H)))))
cv2.imwrite("offt.jpg",20*np.log(abs(o)+1))# output images in the output folder can be used for refrence
cv2.imwrite("output.jpg",abs(o))# output images in the output folder can be used for refrence