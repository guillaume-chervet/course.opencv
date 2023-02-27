import time

import cv2

from text.deskew import deskew
from text.orientation import orientate
from image import rotate

frame = cv2.imread('specimen.png')

deskew_start = time.time()
angle_deskew, original_copy = deskew(frame)
if original_copy is not None:
    cv2.imshow('deskew_rectangles', cv2.pyrDown(original_copy))
img_deskew_cv = rotate(angle_deskew, frame)
deskew_duration = time.time() - deskew_start
orientate_start = time.time()
img_straighted_cv, angle_orientation, duration_orientation = orientate(img_deskew_cv)
orientate_duration = time.time() - orientate_start

print("angle_deskew: ", angle_deskew)

value = [255,255,255]
border = int(1280-720/2)
img_straighted_cv = cv2.pyrDown(img_straighted_cv)
row, col = img_straighted_cv.shape[:2]

border_top = 0
border_left=0
if row > col:
    border_left = int(row-col)
else:
    border_top = int(col-row)
img_straighted_cv = cv2.copyMakeBorder(img_straighted_cv, 0, border_top, 0, border_left, cv2.BORDER_CONSTANT, None, value)
cv2.imshow('frame', img_straighted_cv)
cv2.imshow('orginal', cv2.pyrDown(frame))

cv2.waitKey()
cv2.destroyAllWindows()