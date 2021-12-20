import time

import cv2

from text.deskew import deskew
from text.orientation import orientate
from image import rotate

vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while (True):

    ret, frame = vid.read()

    deskew_start = time.time()
    angle_deskew = deskew(frame)
    img_deskew_cv = rotate(angle_deskew, frame)
    deskew_duration = time.time() - deskew_start
    orientate_start = time.time()
    img_straighted_cv, angle_orientation, duration_orientation = orientate(img_deskew_cv)
    orientate_duration = time.time() - orientate_start

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


