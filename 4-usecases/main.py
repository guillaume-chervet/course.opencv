import cv2
from rectangle_detection import extract_rectangle

original = cv2.imread('guillaume-chervet.png')

extract_rectangle(original, "./output/", environment_mode="development")
