import numpy as np
import cv2

def sliding_window(image,padded_image, mask,window = 3):
    a = np.zeros(image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            a[x][y] = sumQuery(aux, x, y, x+window-1, y+window-1)
    return a

def padding(im,mask):
    image_row, image_col = im.shape
    kernel_row, kernel_col = mask.shape
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = im
    return padded_image

def generate_mask(weight,size = 9):
    mask = (np.ones([size,size]))
    return mask

def preProcess(mat, aux):
    N=aux.shape[1]
    M=aux.shape[0]
    for i in range(0, N, 1): 
        aux[0][i] = mat[0][i] 
    for i in range(1, M, 1): 
        for j in range(0, N, 1): 
            aux[i][j] = mat[i][j] + aux[i - 1][j] 
    for i in range(0, M, 1): 
        for j in range(1, N, 1): 
            aux[i][j] += aux[i][j - 1] 

def sumQuery(aux, tli, tlj, rbi, rbj): 
    res = aux[rbi][rbj]
    if (tli > 0):
        res = res - aux[tli - 1][rbj]
    if (tlj > 0): 
        res = res - aux[rbi][tlj - 1] 
    if (tli > 0 and tlj > 0):
        res = res + aux[tli - 1][tlj - 1]
    return res

im = cv2.imread('image2.jpeg')# path needs to be channged all input images are available in the input folder
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
mask = generate_mask (1,5)
padded_image = padding(im,mask)
aux = np.zeros(padded_image.shape)
preProcess(padded_image,aux)
output = sliding_window(im,aux,mask,5)
output=output/25
cv2.imwrite('imagefastlpf.jpg',output)# output images in the output folder can be used for refrence