import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_histogram(image,i):
    plt.subplot(3,1,i)
    count, bins, patches = plt.hist(image.ravel(), bins=256)
    plt.title("Image "+str(i))
    return count

def hist_equalize(org, spec):
    oldshape = org.shape
    org = org.reshape(-1)
    spec = spec.reshape(-1)
    rand1, bin_idx, s_counts = np.unique(org, return_inverse=True,return_counts=True)
    rand2, t_counts = np.unique(spec, return_counts=True)
    s_quant = np.cumsum(s_counts)
    s_quant = s_quant/s_quant[-1]
    t_quant = np.cumsum(t_counts)
    t_quant = t_quant/t_quant[-1]
    our = s_quant*255
    our = our.astype(int)
    tmp = t_quant*255
    tmp = tmp.astype(int)
    b = []
    for data in our[:]:
        diff = tmp - data
        mask = np.ma.less_equal(diff, -1)
        if np.all(mask):
            c = np.abs(diff).argmin()
        masked_diff = np.ma.masked_array(diff, mask)
        b.append(masked_diff.argmin())
    b = np.array(b,dtype='uint8')
    return b[bin_idx].reshape(oldshape)

spec = cv2.imread('dark.png')# path needs to be channged all input images are available in the input folder
spec = cv2.cvtColor(spec, cv2.COLOR_BGR2GRAY)
org = cv2.imread('mid.jpg')# path needs to be channged all input images are available in the input folder
org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)

a = hist_equalize(org, spec)
get_histogram(org,1)
get_histogram(spec,2)
get_histogram(a,3)

cv2.imwrite("similar.png",a)# output images in the output folder can be used for refrence

# cv2.imwrite("a1.png",org)
# cv2.imwrite("a2.png",spec)
plt.show()