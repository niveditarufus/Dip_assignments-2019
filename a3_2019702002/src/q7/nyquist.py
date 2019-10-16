from skimage.measure import block_reduce
import numpy as np
from numpy import abs, log, rot90, hstack, fft, float, mean, ones, zeros, uint8
from numpy.fft import fftshift, fft2, ifft2, ifftshift
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('bricks.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite("input.jpg",im)
sampled_1 = block_reduce(im, (1, 1))
sampled_2 = block_reduce(im, (2, 2))
sampled_4 = block_reduce(im, (4, 4))
sampled_8 = block_reduce(im, (8, 8))
sampled_16 = block_reduce(im, (16, 16))

sampled1_fft = fftshift(fft2(sampled_1))
sampled1_mag = abs(sampled1_fft)
res1 = ifft2(ifftshift(sampled1_fft)).astype("uint8")
cv2.imwrite("res1.jpg",res1)
cv2.imwrite("fft1.jpg",10*log(sampled1_mag +1))


sampled2_fft = fftshift(fft2(sampled_2))
sampled2_mag = abs(sampled2_fft)
res2 = ifft2(ifftshift(sampled2_fft)).astype("uint8")
cv2.imwrite("res2.jpg",res2)
cv2.imwrite("fft2.jpg",10*log(sampled2_mag +1))

sampled4_fft = fftshift(fft2(sampled_4))
sampled4_mag = abs(sampled4_fft)
res4 = ifft2(ifftshift(sampled4_fft)).astype("uint8")
cv2.imwrite("res4.jpg",res4)
cv2.imwrite("fft4.jpg",10*log(sampled4_mag +1))

sampled8_fft = fftshift(fft2(sampled_8))
sampled8_mag = abs(sampled8_fft)
res8 = ifft2(ifftshift(sampled8_fft)).astype("uint8")
cv2.imwrite("res8.jpg",res8)
cv2.imwrite("fft8.jpg",10*log(sampled8_mag +1))

sampled16_fft = fftshift(fft2(sampled_16))
sampled16_mag = abs(sampled16_fft)
res16 = ifft2(ifftshift(sampled16_fft)).astype("uint")
cv2.imwrite("res16.jpg",res16)
cv2.imwrite("fft16.jpg",10*log(sampled16_mag +1))

# plt.figure()
# plt.imshow(sampled_1, cmap="gray")
# plt.show()

# plt.figure()
# plt.imshow(sampled_2, cmap="gray")
# plt.show()

# plt.figure()
# plt.imshow(sampled_4, cmap="gray")
# plt.show()

# plt.figure()
# plt.imshow(sampled_8, cmap="gray")
# plt.show()

# plt.figure()
# plt.imshow(sampled_16, cmap="gray")
# plt.show()
