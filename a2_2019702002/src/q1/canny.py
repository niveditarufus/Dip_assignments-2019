from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import cv2

im = cv2.imread("noisy.jpg")
edges = cv2.Canny(im,50,100)


cv2.imwrite('barbara_noisy100_200.jpg',edges)
