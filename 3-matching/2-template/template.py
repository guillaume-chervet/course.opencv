import cv2 as cv
import numpy as np


def apply_template(img_rgb, template):
    if len(img_rgb.shape) == 3:
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray, template,cv.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    return img_rgb

template = cv.imread("templates/query.png",0)
img_rgb = cv.imread('templates/modeledelettre.jpg')

apply_template(img_rgb, template)

cv.imshow('Detected Point' , img_rgb)
cv.waitKey()
cv.destroyAllWindows()