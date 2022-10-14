import numpy as np
import cv2

img = np.zeros([720, 1280], dtype=np.uint8)
for y in range(img.shape[0]):
  for x in range(img.shape[1]):
    img[y][x] = 0

cv2.imwrite('accretion-plane.png', img)