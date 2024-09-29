import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image") 
ap.add_argument("-o", "--output", required=False, help="Name to the output") 
ap.add_argument("-b", "--low", required=True, help="a_low value")   # Valor porcentual de a_low
ap.add_argument("-a", "--high", required=True, help="a_high value")  # Valor porcentual de a_high
ap.add_argument("-m", "--min", required=True, help="a_min value")   # Valor a min
ap.add_argument("-n", "--max", required=True, help="a_max value")   # Valor a max
arg = ap.parse_args()
args = vars(arg) 

# Valores para ajustar el contraste
alow = float(args['low'])
ahigh = float(args['high'])

# Valores del nuevo rango de contraste
amin = float(args['min'])
amax = float(args['max'])

image = cv2.imread(args['image'])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histogram is computed
hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
# Histograma normalizado
#hist /= hist.sum()
# Histograma acumulado
cumulative_hist = np.cumsum(hist)

# Obtenemos las dimensiones de la imagen
(M, N) = image_gray.shape

# Obtenemos los valores de las condiciones para a'low y a'high
multlow = int(M*N*alow)
multhigh = int(M*N*(1-ahigh))

# Obtenemos a'low y a'high  (Rango de contraste restringido)
alowp = min([i for i in range(256) if cumulative_hist[i] >= multlow])
ahighp = max([i for i in range(256) if cumulative_hist[i] <= multhigh])

dx = (amax - amin)/(ahighp - alowp)

# Definimos la función lambda para el mapeo
table_map = lambda i: amin if i <= alowp else amax if i >= ahighp else amin + ((i - alowp) * dx)

# Construimos un arreglo para almacenar los valores precaclulados
y2 = np.array([table_map(i) for i in range(256)], dtype='uint8')

# Definir la función lambda para el autocontraste restringido
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
