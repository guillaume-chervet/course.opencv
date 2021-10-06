import cv2
from rectangle_detection import extract_rectangle

original = cv2.imread('sample_anonyme.png')

destination_path = "./output/"
rectangle_img, homography_points = extract_rectangle(original, destination_path, environment_mode="development")

cv2.imwrite(destination_path + "6_img_rectangle.png", rectangle_img)
