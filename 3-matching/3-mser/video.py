import mser

import cv2

#vid = cv2.VideoCapture(0)
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while (True):

    ret, frame = vid.read()

    img_mser = mser.apply_mser(frame)

    cv2.imshow('mser', cv2.pyrDown(img_mser))
    cv2.imshow('orginal', cv2.pyrDown(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


