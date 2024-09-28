import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image") 
ap.add_argument("-b", "--low", required=True, help="q_low value")
ap.add_argument("-a", "--high", required=True, help="q_high value")
# ap.add_argument("-o", "--output", required=True, help="Name to the output") 
arg = ap.parse_args()
args = vars(arg) 

qlow = float(args['low'])
qhigh = float(args['high'])

image = cv2.imread(args['image'])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histograma e histograma acumulativo de la imagen
hist = cv2.calcHist([image], [0], None, [256], [0,256])
histacum = np.cumsum(hist) 


# Obtener el tamaño de la imagen
img_size = image_gray.size

def min(size, histacum, qlow):
    for i in range(256):
        if histacum[i] >= size * qlow:
            return i


def max(size, histacum, qhigh):
    for i in range(256):
        if histacum[i] >= size * (1 - qhigh):
            return i

a_low_prima = min(img_size, histacum, qlow)
a_high_prima = max(img_size, histacum, qhigh)

transform_pixel = lambda pixel: 0 if pixel <= a_low_prima else (255 if pixel >= a_high_prima else (pixel - a_low_prima) * 255 / (a_high_prima - a_low_prima)) 

image_gray_final = np.array(np.vectorize(transform_pixel)(image_gray), dtype='uint8')


fig = plt.figure(figsize=(14,10))
fig.suptitle('Autocontraste restringido')
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.imshow(image_gray, cmap='gray')
ax1.set_title('Imagen original')

ax2.imshow(image_gray_final, cmap='gray')
ax2.set_title('Imagen con autocontraste')

plt.show()
