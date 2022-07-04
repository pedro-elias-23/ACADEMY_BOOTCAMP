import cv2
import numpy as np
import mahotas
from matplotlib import pyplot as plt


img = cv2.imread('ponte.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##Sobel##
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobel = cv2.bitwise_or(sobelX, sobelY)
resultado = np.vstack([np.hstack([img,sobelX]),  np.hstack([sobelY, sobel])  ])
cv2.imshow("Sobel", resultado)
cv2.waitKey(0)

##Filtro Laplaciano##

lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
resultado = np.vstack([img, lap])
cv2.imshow("Filtro Laplaciano", resultado)
cv2.waitKey(0)

##Detector de bordas Canny  ##

suave = cv2.GaussianBlur(img, (7, 7), 0)
canny1 = cv2.Canny(suave, 20, 120)
canny2 = cv2.Canny(suave, 70, 200)
resultado = np.vstack([np.hstack([img,    suave ]),  np.hstack([canny1, canny2])  ])
cv2.imshow("Detector de Bordas Canny", resultado)
cv2.waitKey(0)

###Binarização com limiar###

suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
(T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
(T, binI) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)
resultado = np.vstack([np.hstack([suave, bin]),  np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)])])
cv2.imshow("Binarização da imagem", resultado)
cv2.waitKey(0)

##Threshold adaptativo##

suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
bin1 = cv2.adaptiveThreshold(suave, 255,  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
bin2 = cv2.adaptiveThreshold(suave, 255,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21, 5)
resultado = np.vstack([np.hstack([img, suave]),  np.hstack([bin1, bin2])])
cv2.imshow("Binarização adaptativa da imagem", resultado)
cv2.waitKey(0)

##Threshold com Otsu e Riddler-Calvard##

suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
T = mahotas.thresholding.otsu(suave)
temp = img.copy()
temp[temp > T] = 255
temp[temp < 255] = 0
temp = cv2.bitwise_not(temp)
T = mahotas.thresholding.rc(suave)
temp2 = img.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)
resultado = np.vstack([np.hstack([img, suave]),  np.hstack([temp, temp2])  ])
cv2.imshow("Binarização com método Otsu e Riddler-Calvard", resultado)
cv2.waitKey(0)
