import os
import time
from os import listdir
from os.path import isfile, join
from image import normalize_size

import cv2
from rectangle_detection import extract_rectangle

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DIRECTORY_INPUT_PATH = os.path.join(BASE_PATH, "input")
DIRECTORY_OUTPUT_PATH = os.path.join(BASE_PATH, "output")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

files = [f for f in listdir(DIRECTORY_INPUT_PATH) if isfile(join(DIRECTORY_INPUT_PATH, f))]

for filename in files:
    img = cv2.imread(os.path.join(BASE_PATH, "input", filename))
    img_resized, ratio = normalize_size(img, 1200)
    destination_path = os.path.join(OUTPUT_PATH, filename.split(".")[0])
    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)
    cv2.imwrite(destination_path + "/0_img_origin.png", img_resized)
    start = time.time()
    rectangle_img, homography_points = extract_rectangle(img, destination_path + "/", environment_mode="development")
    end = time.time()
    print(filename)
    print("time: " + str(end - start))
    cv2.imwrite(destination_path + "/7_img_rectangle.png", rectangle_img)

