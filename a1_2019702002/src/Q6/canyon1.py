import cv2
import numpy as np
from matplotlib import pyplot as plt

def gamma_corr(img,gamma_value=1):
	image = np.array(255*(img/255)**gamma_value,dtype='uint8')

	return image

def get_histogram(image,i=1):
    plt.subplot(3,1,i)
    count, bins, patches = plt.hist(image.ravel(), bins=256)
    plt.title("Image "+str(i))
    return count
def linContrastStretching(im, a=0, b=255):
	ahigh = 220
	alow = 55
	get_histogram(im)
	plt.show()

	for x in range(im.shape[0]):
		for y in range(im.shape[1]):
			im[x][y] =  ((im[x][y] - alow)*(b-a)/(ahigh - alow))
	return im

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

spec = cv2.imread('canyon_02_02.png')# path needs to be channged all input images are available in the input folder
spec = cv2.cvtColor(spec, cv2.COLOR_BGR2GRAY)
cv2.imwrite("canyon4ref.png",spec)# output images in the output folder can be used for refrence


org = cv2.imread('part4.png')# path needs to be channged all input images are available in the input folder
org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)

a = hist_equalize(org, spec)
# a = gamma_corr(a,0.4)
# get_histogram(org,1)
# get_histogram(spec,2)
# get_histogram(a,3)
# a = linContrastStretching(a,0,255)
cv2.imwrite("canyon4.png",a)# output images in the output folder can be used for refrence

# plt.show()