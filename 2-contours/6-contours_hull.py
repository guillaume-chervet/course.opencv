import cv2
import numpy as np

img = cv2.pyrDown(cv2.imread("shape.jpg", cv2.IMREAD_UNCHANGED))

ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                            127, 255, cv2.THRESH_BINARY)

contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
black = np.zeros_like(img)
for contour in contours:
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(contour)
    green = (0, 255, 0)
    cv2.drawContours(black, [contour], -1, green, 2)
    blue = (255, 255, 0)
    cv2.drawContours(black, [approx], -1, blue, 2)
    red = (0, 0, 255)
    cv2.drawContours(black, [hull], -1, red, 2)

cv2.imshow("hull", black)
cv2.imshow("original", img)
cv2.waitKey()
cv2.destroyAllWindows()
