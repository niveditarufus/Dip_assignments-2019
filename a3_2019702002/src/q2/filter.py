import numpy as np
import cv2
import math
import cmath

def ideal_lpf(x,D0 = 50):
	H = np.zeros(x.shape)
	for i in range(H.shape[0]):
		for j in range(H.shape[1]):
			y = np.sqrt((i-(H.shape[1])/2)**2 + (j-(H.shape[0])/2)**2)
			if y<=D0:
				H[i,j] = 1
	return H

def butterworth_lpf(x,D0 = 50,n=1):
	H = np.zeros(x.shape)
	for i in range(H.shape[0]):
		for j in range(H.shape[1]):
			y = np.sqrt((i-(H.shape[1])/2)**2 + (j-(H.shape[0])/2)**2)
			H[i,j] = 1/(1 + (y/D0))**(2*n)
	return H

def gaussian(x,D0 = 50):
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

# H = ideal_lpf(im_fft)
# lpf = np.multiply(im_fft,H)
# lpf = np.fft.ifft2(np.fft.fftshift(lpf)).astype("uint8")
# cv2.imwrite("ideal_lpf.jpg",lpf
# output images in the output folder can be used for refrence

# H = butterworth_lpf(im_fft)
# lpf = np.multiply(im_fft,H)
# lpf = np.fft.ifft2(np.fft.fftshift(lpf)).astype("uint8")
# cv2.imwrite("butterworth_lpf.jpg",lpf)
# output images in the output folder can be used for refrence

H = gaussian(im_fft)
lpf = np.multiply(im_fft,H)
lpf = np.fft.ifft2(np.fft.fftshift(lpf)).astype("uint8")
cv2.imwrite("gaussian.jpg",lpf)# output images in the output folder can be used for refrence
