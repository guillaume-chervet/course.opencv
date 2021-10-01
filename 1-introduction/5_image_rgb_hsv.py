import cv2

img = cv2.imread('./sample.png', cv2.IMREAD_COLOR)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('sample-rgb.jpg', img_rgb)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('sample-hsv.jpg', img_hsv)