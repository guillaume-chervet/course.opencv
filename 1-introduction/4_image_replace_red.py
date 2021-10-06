import cv2

img = cv2.imread('./sample_anonyme.png', cv2.IMREAD_COLOR)
img[:, :, 1] = 0
cv2.imshow("sample-color-red", img)
cv2.waitKey()
cv2.destroyAllWindows()
