import cv2
import numpy as np

from .mser import apply_mser


def is_image_contain_text(img):
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img
    mean_color_img = np.mean(gray_img)

    img_mser = apply_mser(img)
    mean_color_mser = np.mean(img_mser)
    distance = abs(mean_color_mser - mean_color_img)
    contain_text = distance < 100
    return contain_text, distance, img_mser

def find_zone_text_img(img):
    original_img = img.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = apply_mser(img)
    ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((6, 6), np.uint8)
    erosion = cv2.dilate(thresh, kernel, iterations=2)
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

    filtered_contours = []
    angles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h * 3 < w < h * 12 and number_larger > number_higher:
            min_rectangle = cv2.minAreaRect(contour)
            angle = min_rectangle[2] % 90
            if angle > 45:
                angle = angle - 90
            if angle < -45:
                angle = angle + 90
            filtered_contours.append({"contour": contour, "angle": angle, "area": w * h})
            angles.append(angle)
        #    box = cv2.boxPoints(min_rectangle)
        #    box = np.int0(box)
        #    red = (0, 0, 255)
        #    cv2.drawContours(img, [box], 0, red, 3)

        elif w * 3 < h < w * 12 and number_larger < number_higher:
            min_rectangle = cv2.minAreaRect(contour)
            angle = min_rectangle[2] % 90
            if angle > 45:
                angle = angle - 90
            if angle < -45:
                angle = angle + 90
            filtered_contours.append({"contour": contour, "angle": angle, "area": w * h})
            angles.append(angle)
          #  box = cv2.boxPoints(min_rectangle)
          #  box = np.int0(box)
          #  red = (0, 0, 255)
          #  cv2.drawContours(img, [box], 0, red, 3)


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
    biggest_contour = None
    if len(filtered_contours) > 0:
        biggest_contour = filtered_contours[0]
        for info in filtered_contours:
            print("area: " + str(info["area"]))
            if biggest_contour["area"] < info["area"]:
                biggest_contour = info

    print("final area: " + str(biggest_contour["area"]))
    if biggest_contour is not None:
        contour = biggest_contour["contour"]
     #   min_rectangle = cv2.minAreaRect(contour)
  #     box = cv2.boxPoints(min_rectangle)
  #     box = np.int0(box)
  #     red = (255, 0, 0)
  #     cv2.drawContours(img, [box], 0, red, 5)
      # return erosion
        x, y, w, h = cv2.boundingRect(contour)
        height, width = img.shape[:2]
        xmin, ymin, xmax, ymax = compute_outer_crop_margin_ratio(width, height, (x, y, x + w, y + h), 2)
        img_crop = original_img[ymin:ymax, xmin: xmax]
        if len(img_crop) > 0:
            return img_crop

    return original_img


def compute_outer_crop_margin_ratio(img_width, img_heigth, coordinates, margin_ratio):
    if margin_ratio <= 0:
        return coordinates
    xmin, ymin, xmax, ymax = coordinates
    margin_ratio = int(max(img_width, img_heigth) * margin_ratio / 100)
    xmin = xmin - margin_ratio
    ymin = ymin - margin_ratio
    xmax = xmax + margin_ratio
    ymax = ymax + margin_ratio
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > img_width:
        xmax = img_width
    if ymax > img_heigth:
        ymax = img_heigth
    return xmin, ymin, xmax, ymax
