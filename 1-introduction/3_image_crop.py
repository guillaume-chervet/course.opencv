import cv2

img = cv2.imread('./sample_anonyme.png')

img_cropped = img[0:200, 0:100]
cv2.imshow("sample-cropped", img_cropped)
cv2.waitKey()
cv2.destroyAllWindows()