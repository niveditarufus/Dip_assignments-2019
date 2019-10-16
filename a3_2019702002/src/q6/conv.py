import numpy as np
from numpy import abs, log, rot90, hstack, fft, float, mean, ones, zeros, uint8
from numpy.fft import fftshift, fft2, ifft2, ifftshift
from scipy.signal import convolve2d
import cv2

f = cv2.imread('lena.jpg')
h = cv2.imread('rectangle.jpg')
f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
h = cv2.cvtColor(h, cv2.COLOR_BGR2GRAY)
M = f.shape[0] + h.shape[0] - 1
N = f.shape[1] + h.shape[1] - 1

conv = convolve2d(f, h.astype(float))
print("fdg")

res = ifft2(ifftshift(fftshift(fft2(f, s=(M, N))) * fftshift(fft2(h, s=(M, N)))))
res = abs(res)
print(h.shape)


print(conv.shape)
cv2.imwrite("idft.jpg",10*np.log(res+1))

cv2.imwrite("conv.jpg",10*np.log(conv+1))

