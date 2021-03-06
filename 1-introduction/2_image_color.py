import cv2

img = cv2.imread('./sample_anonyme.png') #BGR
width = 4
height = 3
dimension = (width, height)
img_resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
cv2.imwrite('sample-small-color.png', img_resized)

print(img_resized)
print(img_resized.shape)
print(img_resized.size)
print(img_resized.dtype)