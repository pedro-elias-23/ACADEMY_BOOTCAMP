import cv2
import numpy as np
import mahotas
from matplotlib import pyplot as plt

img = cv2.imread('ponte.jpg')
img = img[::2,::2] # Diminui a imagem

##Suavização por cálculo da média##

suave = np.vstack([np.hstack([img,cv2.blur(img, ( 3,  3))]),np.hstack([cv2.blur(img, (5,5)), cv2.blur(img, ( 7,  7))]),np.hstack([cv2.blur(img, (9,9)), cv2.blur(img, (11, 11))]),])
cv2.imshow("Imagens suavisadas (Blur)", suave)
cv2.waitKey(0)

##Suavização pela mediana##

suave = np.vstack([np.hstack([img,cv2.medianBlur(img,  3)]),np.hstack([cv2.medianBlur(img,  5),cv2.medianBlur(img,  7)]),np.hstack([cv2.medianBlur(img,  9),cv2.medianBlur(img, 11)]),])
cv2.imshow("Imagem original e suavizadas pela mediana", suave)
cv2.waitKey(0)

##Suavização com filtro bilateral##

suave = np.vstack([np.hstack([img,cv2.bilateralFilter(img,  3, 21, 21)]),np.hstack([cv2.bilateralFilter(img,  5, 35, 35),cv2.bilateralFilter(img,  7, 49, 49)]),np.hstack([cv2.bilateralFilter(img,  9, 63, 63),cv2.bilateralFilter(img, 11, 77, 77)])])
cv2.imshow("Imagem original e Suavização com filtro bilatera", suave)
cv2.waitKey(0)