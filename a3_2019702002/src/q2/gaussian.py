import numpy as np
import cv2
import math
import cmath

def gaussian_filter(x,D0 = 50):
	H = np.zeros(x.shape)
	for i in range(H.shape[0]):
		for j in range(H.shape[1]):
			y = np.sqrt((i-(H.shape[1])/2)**2 + (j-(H.shape[0])/2)**2)
			H[i,j] = math.exp(-((y**2)/(2*(D0)**2)))
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

H = gaussian_filter(im_fft,20)
lpf = np.multiply(im_fft,H)
lpf = np.fft.ifft2(np.fft.fftshift(lpf)).astype("uint8")
cv2.imwrite("gaussian20.jpg",lpf)# output images in the output folder can be used for refrence

# im1 = cv2.imread('gaussian20.jpg')
# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
# im2 = cv2.imread('gaussian60.jpg')
# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
# diff = im2 -im1
# cv2.imwrite("gaussian_diff.jpg",diff)