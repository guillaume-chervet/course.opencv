import cv2
import numpy as np


def nothing(*arg):
    pass


def simple_trackbar(image, window_name):
    trackbar_name = window_name + "Trackbar"

    cv2.namedWindow(window_name)
    cv2.createTrackbar(trackbar_name, window_name, 0, 255, nothing)

    threshold = np.zeros(image.shape, np.uint8)

    while True:
        trackbar_position = cv2.getTrackbarPos(trackbar_name, window_name)
        cv2.threshold(image, trackbar_position, 255, cv2.THRESH_BINARY, threshold)
        cv2.imshow(window_name, threshold)

        # If you press "ESC", it will return value
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()
    return threshold


img_grey = cv2.imread('sample_anonyme.png', cv2.IMREAD_GRAYSCALE)
simple_trackbar(img_grey, "threshold")
