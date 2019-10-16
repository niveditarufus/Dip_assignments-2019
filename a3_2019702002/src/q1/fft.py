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
		print([X_even + factor[:int(N / 2)] * X_odd,X_even + factor[int(N / 2):] * X_odd])
		return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
		                       X_even + factor[int(N / 2):] * X_odd])
		
a = np.array([1,2,3,4])
N = a.shape[0]
y = Fast_fourier(a)
print (y)
