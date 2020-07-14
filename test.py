import numpy as np
import cv2

path = 'images/dog.png'

image = cv2.imread(path)
image = cv2.rectangle(image, (100, 100),  (200, 200), (0, 0, 255), 2)
cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
