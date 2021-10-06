import cv2
import numpy as np

img_grey = cv2.imread('sample_anonyme.png', cv2.IMREAD_GRAYSCALE)
threshold = np.zeros(img_grey.shape, np.uint8)
cv2.threshold(img_grey, 125, 255, cv2.THRESH_BINARY, threshold)
cv2.imshow("threshold", threshold)
cv2.waitKey()
cv2.destroyAllWindows()