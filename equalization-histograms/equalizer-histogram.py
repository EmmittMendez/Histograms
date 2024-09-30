import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-o", "--output", required=False, help="Name to the output")
ap.add_argument("-k", "--k", required=False, help="k value")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
k = float(args['k'])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histogram is computed
hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
# Histograma acumulado
cumulative_hist = np.cumsum(hist)

# Obtenemos las dimensiones de la imagen
(M, N) = image_gray.shape

# Factor de cambio
dx = (k-1)/(M*N)

# Construimos un vector X y Y para almacenar los valores precalculados
y2 = np.array([np.round(cumulative_hist[i] * dx)for i in range(256)], dtype='uint8')

# Definimos una función lambda para mapear los valores
equalizer = lambda m: y2[m]

# Se aplica la función lambda a cada pixel de la imagen. Se vectoriza la
# matriz para que pueda ser procesado por la función lambda
image_equalized = np.array(np.vectorize(equalizer)(image_gray), dtype='uint8')

# Se calcula el histograma de la imagen ecualizada
hist_equalized = cv2.calcHist([image_equalized], [0], None, [256], [0, 256])
# Histograma acumulado
cumulative_equal = np.cumsum(hist_equalized)

image_equalized = cv2.merge([image_equalized, image_equalized, image_equalized])

# Se genera una figura para mostrar la imagen y su histograma
fig = plt.figure(figsize=(9,6))
# Se crean los subplots del grafico
ax1 = fig.add_subplot(2,3,1)    # Imagen original
ax2 = fig.add_subplot(2,3,2)    # Histograma de la imagen original
ax3 = fig.add_subplot(2,3,3)    # Histograma acumulado
ax4 = fig.add_subplot(2,3,4)    # Imagen ecualizada
ax5 = fig.add_subplot(2,3,5)    # Histograma de la imagen ecualizada
ax6 = fig.add_subplot(2,3,6)    # Histograma acumulado de la imagen ecualizada

# Se dibuja la imagen original
ax1.imshow(image_gray, cmap='gray')
ax1.set_title("Original Image")

# Se dibuja el histograma de la imagen original
ax2.plot(hist)
ax2.set_title("Histogram")

# Se dibuja el histograma acumulado de la imagen original
ax3.plot(cumulative_hist)
ax3.set_title("Cumulative Histogram")

# Se dibuja la imagen ecualizada
ax4.imshow(image_equalized, cmap='gray')
ax4.set_title("Equalization")

ax5.plot(hist_equalized)
ax5.set_title("Equalization Histogram")

ax6.plot(cumulative_equal)
ax6.set_title("Cumulative Histogram")

# Guardar la imagen si se especifica un archivo de salida
if args['output']:
    cv2.imwrite(args['output'], image_equalized)
plt.show()