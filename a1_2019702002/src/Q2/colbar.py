from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import cv2 
bgr_image = cv2.imread('contrast_stretched_lena.jpg')
plt.imshow(bgr_image, cmap = 'gray')
plt.colorbar(cmap = 'gray',fraction=0.03, pad=0.04)
plt.show()  