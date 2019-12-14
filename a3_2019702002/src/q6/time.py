import time
from time import time
import numpy as np
from numpy import abs, log, rot90, hstack, fft, float, mean, ones, zeros, uint8
from numpy.fft import fftshift, fft2, ifft2, ifftshift
from scipy.signal import convolve2d
import cv2
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt


def conv_normal(img1, img2):
    return convolve2d(img1, img2.astype(float))


def conv_fft(img1, img2):
    M = img1.shape[0] + img2.shape[0] - 1
    N = img1.shape[1] + img2.shape[1] - 1
    res = ifft2(ifftshift(fftshift(fft2(img1, s=(M, N))) * fftshift(fft2(img2, s=(M, N)))))
    res = abs(res)
    return res


h = cv2.imread('bricks.jpg')# path needs to be channged all input images are available in the input folder
im1 = cv2.cvtColor(h, cv2.COLOR_BGR2GRAY)
im2 = im1[:128,:128]
im3 = im1[:64,:64]
im4 = im1[:32,:32]
im_dict = {0:im4, 1:im3, 2:im2, 3:im1}
print(im_dict)
norm_time = zeros((5,5))
fft_time = zeros((5,5))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for row in range(4):
    for col in range(4):
        start = time()
        res = conv_fft(im_dict[row], im_dict[col])
        end = time()
        fft_time[row][col] = end - start
        f = end - start
        start = time()
        res = conv_normal(im_dict[row], im_dict[col])
        end = time()
        norm_time[row][col] = end - start
        n = end - start
        ax.scatter((2**row)*32, (2**col)*32, f, c='r', marker='o')
        ax.scatter((2**row)*32, (2**col)*32, n, c='b', marker='^')

        
# print(fft_time)
# print(norm_time)


ax.set_xlabel('Image 1 Size')
ax.set_ylabel('Image 2 Size')
ax.set_zlabel('Time')
plt.show()
