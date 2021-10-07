import cv2

img = cv2.imread('./sample_anonyme.png') #BGR

img[:, :, 0] = 0
img[:, :, 1] = 0
cv2.imshow("sample-color-red", img)
cv2.waitKey()
cv2.destroyAllWindows()
