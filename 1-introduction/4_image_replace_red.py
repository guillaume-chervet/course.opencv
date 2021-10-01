import cv2

img = cv2.imread('./sample.png', cv2.IMREAD_COLOR)
img[:, :, 1] = 0
cv2.imwrite('sample-color-red.jpg', img)

