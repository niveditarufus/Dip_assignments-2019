from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import cv2 

def get_frequent_color(image, k=3):
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return list(dominant_color)

bgr_image = cv2.imread('flowers.jpg') # path needs to be channged all input images are available in the input folder
dom_color = get_frequent_color(bgr_image, k=3)
dom_color_bgr = np.full(bgr_image.shape, dom_color, dtype='uint8')
cv2.imwrite('Image Dominant Color.jpg', dom_color_bgr)
