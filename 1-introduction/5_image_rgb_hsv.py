import cv2

img = cv2.imread('./sample_anonyme.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("sample-rgb.jpg", img_rgb)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("sample-hsv.jpg", img_hsv)

cv2.waitKey()
cv2.destroyAllWindows()
