import math
import random

import cv2
import numpy as np


def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False


def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def find_contour(threshold, img2, destination_path="", min_area=5000, max_area=0.20):
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    (h, w) = img2.shape[:2]
    x_middle = int(w /2)
    number_rect_left = 0
    number_rect_right = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            (p0, p1, p2, p3) = box
            line1 = math.sqrt(((p0[0] - p1[0]) ** 2) + ((p0[1] - p1[1]) ** 2))
            line2 = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

            if line1 > line2 and line1 / 30 > line2:
                continue
            elif line2 > line1 and line2 / 30 > line1:
                continue

            origin_area = h * w
            percentage_area_max = int(origin_area * max_area)
            area = line2 * line1
            if area > percentage_area_max:
                continue

            delta_x = p0[0] - p1[0]
            delta_y = p0[1] - p1[1]
            theta_radians = math.atan2(delta_y, delta_x) * 180 / math.pi
            reste = theta_radians % 90
            if (reste < 5) or (reste > 85):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.drawContours(img2, [box], 0, color, 5)

                if p0[0] < x_middle or p1[0] < x_middle or p2[0] < x_middle or p3[0] < x_middle:
                    number_rect_right = number_rect_right + 1

                if p0[0] > x_middle or p1[0] > x_middle or p2[0] > x_middle or p3[0] > x_middle:
                    number_rect_left = number_rect_left + 1
    return img2, number_rect_right, number_rect_left

def sobel(img):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def extract_table(img, destination_path=""):
    img = img.copy()
    img = cv2.bilateralFilter(img, 9, 75, 75)
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = sobel(gray)
    alpha = 3  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY, 15, -2)
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    # [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 28
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # [vert]
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 28
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.bitwise_not(vertical)
    edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel)
    smooth = np.copy(vertical)
    smooth = cv2.blur(smooth, (2, 2))
    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]

    table_segment = cv2.addWeighted(cv2.bitwise_not(vertical), 0.5, horizontal, 0.5, 0.0)
    toto, thresh = cv2.threshold(table_segment, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.dilate(thresh, kernel, 2)
    img_out, number_rect_right, number_rect_left = find_contour(thresh, img.copy(), destination_path=destination_path, min_area=3000, max_area=0.08)
    return thresh, img_out, number_rect_right, number_rect_left
