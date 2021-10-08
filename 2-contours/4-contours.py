import cv2
import numpy as np

img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 255

contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
green = (0, 255, 0)
img = cv2.drawContours(color, contours, -1, green, 2)
cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()
