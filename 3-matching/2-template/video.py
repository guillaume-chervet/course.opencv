import cv2
import numpy as np

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#vid = cv2.VideoCapture(1)
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def apply_template(img_rgb, template):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.64
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return img_rgb

template = cv2.imread("templates/book_mongo_300.jpg", 0)
while (True):
    ret, frame = vid.read()
    img = apply_template(frame, template)
    cv2.imshow('template', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()