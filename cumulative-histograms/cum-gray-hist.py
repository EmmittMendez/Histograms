import matplotlib.pyplot as plt
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)

# Histogram is computed
hist = cv2.calcHist([image], [0], None, [256], [0, 256], accumulate=True)
# Nomalzed histogram
hist /= hist.sum()
# Cumulative histogram
cumulative_hist = hist.cumsum()

# Se genera una figura para mostrar la imagen y su histograma
fig = plt.figure(figsize=(9,6))
# Se crean los subplots del grafico
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
# Se dibuja la imagen original
ax1.imshow(image, cmap='gray')
ax1.set_title("Original Image")

# Se dibuja el histograma
ax2.plot(cumulative_hist)
ax2.set_title("Histogram")

plt.show()