import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('rot1.png',0)
img1 = cv2.resize(img1,(1024,1024))
img2 = cv2.imread('rot2.png',0)
img2 = cv2.resize(img2,(1024,1024))

cx,cy = img1.shape
M = cv2.getRotationMatrix2D((cx/2,cy/2), 30, 1)
# img2 = cv2.warpAffine(img1, M, (cx, cy))

hanw = cv2.createHanningWindow((cx,cy),cv2.CV_64F)
img1 = img1 * hanw
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = 10*np.log(np.abs(fshift1) +1)
cv2.imwrite('log_FFT1.png', magnitude_spectrum1)

M90 = cv2.getRotationMatrix2D((cx/2,cy/2), 90, 1)
polar_map1= cv2.linearPolar(magnitude_spectrum1, (cx/2,cy/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
p1 = cv2.warpAffine(polar_map1, M90, (cx, cy))
cv2.imwrite('polar1.png', p1)


img2 = img2 * hanw
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 = 10*np.log(np.abs(fshift2) +1)
cv2.imwrite('log_FFT2.png', magnitude_spectrum2)
polar_map2= cv2.linearPolar(magnitude_spectrum2, (cx/2,cy/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
p2 = cv2.warpAffine(polar_map2, M90, (cx, cy))
cv2.imwrite('polar2.png', p2)

# cv2.imwrite('input1.png',img1)
# cv2.imwrite('input2.png',img2)

