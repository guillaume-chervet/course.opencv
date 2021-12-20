import cv2
import numpy as np


def apply_mser(img, max_area=300, min_area=6):
    mser = cv2.MSER_create(max_area=max_area, min_area=min_area)

    if len(img.shape) ==2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vis = img.copy()
    regions, mser_bboxes = mser.detectRegions(gray)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    text_only = cv2.bitwise_and(img, img, mask=mask)

    mask = cv2.bitwise_not(mask)
    background = np.full(img.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)

    final = cv2.bitwise_or(text_only, bk)

    return final