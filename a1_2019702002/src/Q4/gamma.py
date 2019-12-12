from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2 
import sys
import math

def gamma_corr(img,gamma_value=1):
	image = np.array(255*(img/255)**gamma_value,dtype='uint8')

	return image

im = cv2.imread('gamma-corr.png')# path needs to be channged all input images are available in the input folder

img = gamma_corr(im,1.5)

cv2.imwrite("gamma = 1.5.jpg",img)# output images in the output folder can be used for refrence

