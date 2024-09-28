import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image") 
ap.add_argument("-o", "--output", required=False, help="Name to the output") 
ap.add_argument("-b", "--min", required=True, help="a_min value")
ap.add_argument("-a", "--max", required=True, help="a_max value")
arg = ap.parse_args()
args = vars(arg) 

# Valores para ajustar el contraste
amin = float(args['min'])
amax = float(args['max'])

image = cv2.imread(args['image'])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Obtenemos el valor máximo y mínimo de la imagen
ahigh = image_gray.max()
alow = image_gray.min()

# Valor de cambio del contraste
dx = (amax - amin)/(ahigh - alow)

# Construimos un vector X y Y para almacenar los valores precaclulados
x = np.linspace(0, 255,256)
y2 = np.array([amin + (i-alow)*dx for i in range(256)], dtype='uint8')

# Definimos una función lambda para el autocontraste
contrast = lambda m: y2[m]

# Se aplica la función lambda a cada pixel de la imagen. Se vectoriza la
# matriz para que pueda ser procesado por la función lambda
image_contrast = np.array(np.vectorize(contrast)(image_gray), dtype='uint8')

# Histogram is computed
hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
hist_contrast = cv2.calcHist([image_contrast], [0], None, [256], [0, 256])

fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)    # Histograma de la imagen original
ax4 = fig.add_subplot(2,2,4)    # Histograma de la imagen con contraste

ax1.imshow(image_gray, cmap='gray')
ax1.set_title("Gray Image")

ax2.imshow(image_contrast, cmap='gray')
ax2.set_title("Autocontrast Image")


ax3.plot(hist)
ax3.set_title("Histogram")
ax4.plot(hist_contrast)
ax4.set_title("Autocontrast Histogram")

#cv2.imwrite(args['output'], image_contrast)

plt.show()
