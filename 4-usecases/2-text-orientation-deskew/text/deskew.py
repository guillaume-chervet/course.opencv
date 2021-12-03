import cv2
import numpy as np


def deskew(mser_img):
    img = mser_img
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((6, 6), np.uint8)
    erosion = cv2.dilate(thresh, kernel, iterations=1)
    contours, hier = cv2.findContours(erosion, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    number_larger = 0
    number_higher = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > h:
            number_larger = number_larger + 1
        else:
            number_higher = number_higher + 1

    is_rotated = False
    if number_higher > number_larger:
        erosion = cv2.rotate(erosion, cv2.ROTATE_90_CLOCKWISE)
        contours, hier = cv2.findContours(erosion, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        is_rotated = True

    filtered_contours = []
    angles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h * 3 < w < h * 16:
            filtered_contours.append(contour)
            min_rectangle = cv2.minAreaRect(contour)
            angle = min_rectangle[2] % 90
            if angle > 45:
                angle = angle - 90
            if angle < -45:
                angle = angle + 90
            angles.append(angle)

    if len(angles) == 0:
        return 0

    sorted_angles = np.sort(angles)
    previous_angle = None
    new_array = []
    for angle in sorted_angles:
        if previous_angle is not None:
            new_array.append(angle - previous_angle)
        previous_angle = angle

    threshod = np.mean(np.array(new_array)) * 2

    previous_angle = None
    final_array = [[]]
    index = 0
    for angle in sorted_angles:
        if previous_angle is not None:
            if angle - previous_angle > threshod:
                index = index + 1
                final_array.append([])
        previous_angle = angle
        final_array[index].append(angle)

    max_array_lenght = []
    for array in final_array:
        if len(max_array_lenght) < len(array):
            max_array_lenght = array

    angle_mean = np.mean(np.array(max_array_lenght))

    if is_rotated:
        angle_mean = angle_mean + 90

    return round(angle_mean, 2)
