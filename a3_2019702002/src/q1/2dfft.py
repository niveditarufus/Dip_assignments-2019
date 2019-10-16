import numpy as np
import cv2
import math
import cmath
def Fast_fourier(x):
	N = x.shape[0]
	if N ==1:
		return x[0]
	else:
		X_even = Fast_fourier(x[::2])
		X_odd = Fast_fourier(x[1::2])
		factor = np.exp(-2*complex(0,1) * np.pi * np.arange(N) / N)
		return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
		                       X_even + factor[int(N / 2):] * X_odd])
def get_fourier_2D(x):
	y = np.zeros(x.shape)
	for i in range(x.shape[0]):
		y[i] = Fast_fourier(x[:,i])
	z = np.zeros(x.shape)
	for i in range(x.shape[1]):
		z[i] = Fast_fourier(y[:,i])
	return z

im = cv2.imread('rectangle.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
fimg = get_fourier_2D(im)
im_fft = np.fft.fftshift(fimg)
mag = 10*np.log(abs(im_fft) + 1)
cv2.imwrite("test1.jpg",mag)