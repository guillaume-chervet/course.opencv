import cv2

img = cv2.imread('./sample.png', cv2.IMREAD_COLOR)

img_cropped = img[0:200, 0:100]
cv2.imwrite('sample-cropped.jpg', img_cropped)


