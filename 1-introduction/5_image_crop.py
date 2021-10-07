import cv2

img = cv2.imread('./sample_anonyme.png') #BGR

img_cropped = img[0:400, 0:800]
cv2.imshow("sample-cropped", img_cropped)
cv2.imshow("orginal", img)
cv2.waitKey()
cv2.destroyAllWindows()