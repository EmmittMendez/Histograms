import matplotlib.pyplot as plt
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(image_RGB)

# Histogram is computed
histr = cv2.calcHist([r], [0], None, [256], [0, 256])
histg = cv2.calcHist([g], [0], None, [256], [0, 256])
histb = cv2.calcHist([b], [0], None, [256], [0, 256])

# Normalized histograms
histr /= histr.sum()
histg /= histg.sum()
histb /= histb.sum()

# Se genera una figura para mostrar la imagen y su histograma
fig = plt.figure(figsize=(12,9))
# Se crean los subplots del grafico
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,3,4)
ax3 = fig.add_subplot(2,3,5)
ax4 = fig.add_subplot(2,3,6)

# Se dibuja la imagen original
ax1.imshow(image_RGB)
ax1.set_title("Original Image")

# Se dibuja el histograma
ax2.plot(histr, color='r')
ax2.set_title("Histogram of Red Channel")

ax3.plot(histg, color='g')
ax3.set_title("Histogram of Green Channel")

ax4.plot(histb, color='b')
ax4.set_title("Histogram of Blue Channel")

plt.show()