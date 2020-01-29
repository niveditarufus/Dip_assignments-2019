import numpy as np
import cv2

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

im = cv2.imread("figure1.png")
r = rotateImage(im,30)
cv2.imwrite('figure2.png',r)