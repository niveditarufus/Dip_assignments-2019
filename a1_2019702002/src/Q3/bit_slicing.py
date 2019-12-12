from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

def bit_plane(image,bits=8):
	plane = np.empty([image.shape[0],image.shape[1],bits])
	for x in range(bits):
		plane[:,:,x] = (image/math.pow(2,x))%2
	return plane
		

image = cv2.imread('cameraman.png')# path needs to be channged all input images are available in the input folder
grey_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
planes = bit_plane(grey_im)
fig, ax = plt.subplots(3,3)
ax[0][0].imshow(grey_im,cmap = 'gray')
ax[0][0].title.set_text("original")
print((planes.shape[2]))
ax[0][1].imshow(planes[:,:,0],cmap = 'gray')
ax[0][1].title.set_text("bit 1 set")

ax[0][2].imshow(planes[:,:,1],cmap = 'gray')
ax[0][2].title.set_text("bit 2 set")

ax[1][0].imshow(planes[:,:,2],cmap = 'gray')
ax[1][0].title.set_text("bit 3 set")

ax[1][1].imshow(planes[:,:,3],cmap = 'gray')
ax[1][1].title.set_text("bit 4 set")

ax[1][2].imshow(planes[:,:,4],cmap = 'gray')
ax[1][2].title.set_text("bit 5 set")

ax[2][0].imshow(planes[:,:,5],cmap = 'gray')
ax[2][0].title.set_text("bit 6 set")

ax[2][1].imshow(planes[:,:,6],cmap = 'gray')
ax[2][1].title.set_text("bit 7 set")

ax[2][2].imshow(planes[:,:,7],cmap = 'gray')
ax[2][2].title.set_text("bit 8 set")






plt.show()

