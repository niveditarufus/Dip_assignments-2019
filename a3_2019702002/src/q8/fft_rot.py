import numpy as np
from numpy import abs, log, rot90, hstack, fft, float, mean, ones, zeros, uint8
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import cv2

im = cv2.imread('rectangle.jpg')# path needs to be channged all input images are available in the input folder
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

fft = fft2(im)
fft = fftshift(fft)
mag = abs(fft)
log_mag = 20*log(mag + 1)
cv2.imwrite("normal.jpg", log_mag)

# abs(fft2(rot90(fft2(pattern_input), 2)))
rot = rot90(im)
cv2.imwrite("rot.jpg",rot)# output images in the output folder can be used for refrence

fft = fft2(rot)
fft = fftshift(fft)
mag = abs(fft)
log_mag = 20*log(mag + 1)
cv2.imwrite("rotfft.jpg", log_mag)# output images in the output folder can be used for refrence


translation_matrix = np.float32([ [1,0,70], [0,1,70] ])
trans = cv2.warpAffine(im, translation_matrix, (im.shape))
cv2.imwrite("trans.jpg",trans)# output images in the output folder can be used for refrence

fft = fft2(trans)
fft = fftshift(fft)
mag = abs(fft)
log_mag = 20*log(mag + 1)
cv2.imwrite("transfft.jpg", log_mag)# output images in the output folder can be used for refrence
