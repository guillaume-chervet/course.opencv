import numpy as np
import cv2
from image import image_resize

original = cv2.imread('sample_anonyme.png')
resized_original, ratio = image_resize(original, 400)
img = resized_original.copy()
height, width, channel = img.shape

# Grabcut
rectangle_for_grabcut = (12, 12, width - 12, height - 12)
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(img, mask, rectangle_for_grabcut, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

# Draw rectangle
start_point = (rectangle_for_grabcut[0], rectangle_for_grabcut[1])
end_point = (rectangle_for_grabcut[2], rectangle_for_grabcut[3])
red = (0, 0, 255)
thickness = 2
image = cv2.rectangle(img, start_point, end_point, red, thickness)

cv2.imshow("grabcut", img)
cv2.imshow("original", resized_original)
cv2.waitKey()
cv2.destroyAllWindows()
