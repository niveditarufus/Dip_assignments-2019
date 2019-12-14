import numpy as np
import cv2
import math
import cmath
import scipy.signal
import matplotlib.pyplot as plt

def padding(im,kernel_row = 3, kernel_col = 3):
	image_row, image_col = im.shape
	 
	pad_height = int((kernel_row - 1) / 2)
	pad_width = int((kernel_col - 1) / 2)
	 
	padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
	 
	padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = im
	return padded_image
def butterworth_BR(x,D0 = 73,w=50,n=3):
	H = np.ones(x.shape)
	for i in range(H.shape[0]):
		for j in range(H.shape[1]):
			D = np.sqrt((i-(H.shape[0])/2)**2 + (j-(H.shape[1])/2)**2)
			y = ((D*w)/(D**2 - D0**2))**(2*n)
			H[i,j] = 1/(1 + y)

	cv2.imwrite('filter.jpg',20*np.log(abs(H)+1))# output images in the output folder can be used for refrence
	return H

def filter(x,D0,n=1):
	H = np.zeros(x.shape)
	H1 =H
	for i in range(H.shape[0]):
		for j in range(H.shape[1]):
			y = np.sqrt((i-(H.shape[0])/2)**2 + (j-(H.shape[1])/2)**2)
			H[i,j] = 1/(1 + (y/D0))**(2*n)
			# if y<=D0:
				# H[i,j] = 1
			# H[i,j] = 1 - H[i,j]
	# H = np.fft.fftshift(H)

	cv2.imwrite('filterlpf.jpg',20*np.log(abs(H)+1))# output images in the output folder can be used for refrence

	return H


im = cv2.imread('land.png')# path needs to be channged all input images are available in the input folder
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im = padding(im)
im_fft = np.fft.fft2(im)
im_fft = np.fft.fftshift(im_fft)
mag = 10*np.log(abs(im_fft) + 1)
cv2.imwrite("fft.jpg", mag)# output images in the output folder can be used for refrence
# H1 = notchfilter(im_fft,10,46,78,1)
# H2 = notchfilter(im_fft,10,46,182,1)
# H3 = notchfilter(im_fft,10,150,78,1)
# H4 = notchfilter(im_fft,10,150,182,1)
# H5 = notchfilter(im_fft,10,25,129,1)
# H6 = notchfilter(im_fft,10,171,129,1)
# H7 = notchfilter(im_fft,10,98,58,1)
# H8 = notchfilter(im_fft,10,98,213,1)

# H1 = notchfilter(im_fft,10, 80, 135,1)
# H2 = notchfilter(im_fft,10, 160, 76 ,1)
# H3 = notchfilter(im_fft,10,78, 170,1)
# H4 = notchfilter(im_fft,10,177, 149,1)
# H5 = notchfilter(im_fft,10,184, 160,1)
# H6 = notchfilter(im_fft,10,177, 184,1)
H = butterworth_BR(im_fft)
H1 = butterworth_BR(im_fft,134,50)
H = np.multiply(H1,H)
# H = np.multiply(H,H3)
# H = np.multiply(H,H4)
# H = np.multiply(H,H5)
# H = np.multiply(H,H6)
# H = np.multiply(H,H7)
# H = np.multiply(H,H8)
output = np.multiply(im_fft,H)
# output = np.multiply(output,H1)
# H1 = butterworth_BR(im_fft,50,5)
# output = np.multiply(output,H1)

mag = 10*np.log(abs(output) + 1)
cv2.imwrite("ffto.jpg", mag)# output images in the output folder can be used for refrence
output = np.fft.ifft2(np.fft.fftshift(output)).astype("uint8")
cv2.imwrite("output.jpg",output)# output images in the output folder can be used for refrence
plt.imshow(mag,cmap = 'gray')
plt.show()