import cv2
import numpy
import sys

def mergeImage(f_image,b_image):
	height, width, channels = f_image.shape
	a1 = .01
	a2 = 2
	alpha = numpy.ones((height, width))
	for x in range(0,height):
		for y in range(0,width):
			pixel = f_image[x,y]
	
			B = pixel[0]
			G = pixel[1]
			
			alpha[x,y] = 1-(a1*(G-(a2*B)))
	
			if alpha[x,y]<0:
				alpha[x,y]=0
			if alpha[x,y]>1:
				alpha[x,y]=1
			
			f_image[x,y] = alpha[x,y]*f_image[x,y]
			
			if alpha[x,y]==0:
				f_image[x,y] = b_image[x,y]
	return f_image

# path_image=sys.argv[1]
# background = sys.argv[2]
f = cv2.imread('fg.jpg')
b = cv2.imread('bg.jpg')
f = mergeImage(f,b)
cv2.imwrite( "result.jpg", f)


