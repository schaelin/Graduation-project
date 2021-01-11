import cv2
import matplotlib.pyplot as plt

originImg = cv2.imread('Lenna.png', cv2.IMREAD_COLOR)
grayImg = cv2.cvtColor(originImg, cv2.COLOR_BGR2GRAY)

plt.hist(grayImg.ravel(), 256, [0, 256])
plt.show()

cv2.imshow('Image', grayImg)
cv2.waitKey()