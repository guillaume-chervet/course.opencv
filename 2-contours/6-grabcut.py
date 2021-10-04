import numpy as np
import cv2

original = cv2.imread('statue_small.jpg')
img = original.copy()
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (100, 1, 421, 378)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

# Draw rectangle
start_point = (100, 1)
end_point = (421, 378)
color = (0, 0, 255)
thickness = 2
image = cv2.rectangle(img, start_point, end_point, color, thickness)

cv2.imshow("grabcut", img)
cv2.imshow("original", original)
cv2.waitKey()
cv2.destroyAllWindows()
