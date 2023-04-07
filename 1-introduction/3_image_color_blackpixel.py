import cv2

img = cv2.imread('./sample_anonyme.png') #BGR
width = 4
height = 3
dimension = (width, height)
img_resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
img_resized[0, 0] = [0, 0, 0] #[Blue, Green, Red]
print(img_resized)
cv2.imwrite('sample-small-color-blackpixel.png', img_resized)
