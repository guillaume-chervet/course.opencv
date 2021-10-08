import cv2
import numpy as np

img = cv2.pyrDown(cv2.imread("hammer.jpg", cv2.IMREAD_UNCHANGED))

ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                            127, 255, cv2.THRESH_BINARY)

contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # find bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
    green = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), green, 2)

    # find minimum area
    min_rectangle = cv2.minAreaRect(contour)
    box = cv2.boxPoints(min_rectangle)
    box = np.int0(box)
    red = (0, 0, 255)
    cv2.drawContours(img, [box], 0, red, 3)

blue = (255, 0, 0)
cv2.drawContours(img, contours, -1, blue, 2)
cv2.imshow("contours", img)

cv2.waitKey()
cv2.destroyAllWindows()
