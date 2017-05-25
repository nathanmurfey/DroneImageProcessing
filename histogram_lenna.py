import cv2
import numpy
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(122)
ay = fig.add_subplot(121)
lenna = cv2.imread('TestImages/Lenna.png')
gray_lenna = cv2.cvtColor(lenna, cv2.COLOR_RGB2GRAY)
hist_full = cv2.calcHist([lenna], [0], None, [256], [0, 256])
cv2.imshow("lenna", gray_lenna)
ay.imshow(gray_lenna,cmap='Greys_r')
ax.plot(hist_full)
plt.show()

#imshow(object_detected,cmap='Greys_r'