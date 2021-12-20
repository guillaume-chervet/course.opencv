import cv2
from table import extract_table

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while (True):

    ret, frame = vid.read()

    thresh, img_out, number_rect_right, number_rect_left = extract_table(frame)
    cv2.imshow('rectangles', img_out)
    cv2.imshow('thresh', thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()