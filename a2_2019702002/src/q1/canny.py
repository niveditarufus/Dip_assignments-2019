from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import cv2

im = cv2.imread("noisy.jpg")# path needs to be channged all input images are available in the input folder
edges = cv2.Canny(im,50,100)


cv2.imwrite('barbara_noisy100_200.jpg',edges)# output images in the output folder can be used for refrence
