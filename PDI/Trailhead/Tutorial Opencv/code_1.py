# Importação das bibliotecas
import cv2
import numpy as np

img = cv2.imread('ponte.jpg')
(b, g, r) = img[0, 0] #veja que a ordem BGR e não RGB

print('O pixel (0, 0) tem as seguintes cores:')
print('Vermelho:', r, 'Verde:', g, 'Azul:', b)


for y in range(0, img.shape[0]):
   for x in range(0, img.shape[1]):
     img[y, x] = (255,0,0)
cv2.imshow("Imagem modificada", img)
cv2.waitKey(0)

for y in range(0, img.shape[0], 10): #percorre linhas
   for x in range(0, img.shape[1], 10): #percorre colunas
     img[y:y+5, x: x+5] = (0,255,255)
cv2.imshow("Imagem modificada", img)
cv2.waitKey(0)

##Cortando uma imagem / Crop##

recorte = img[100:200, 100:200]
cv2.imshow("Recorte da imagem", recorte)
cv2.imwrite("recorte.jpg", recorte) #salva no disco
