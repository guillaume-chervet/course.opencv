import cv2
import numpy as np

def nothing(*arg):
    pass

def simple_trackbar(image, window_name):

    cv2.namedWindow(window_name)
    cv2.createTrackbar("threshold", window_name, 0, 255, nothing)
    cv2.createTrackbar("erode", window_name, 0, 30, nothing)
    cv2.createTrackbar("dilate", window_name, 0, 30, nothing)
    cv2.createTrackbar("color", window_name, 0, 1, nothing)
    #cv2.createTrackbar("type", window_name, 0, 2, nothing)

    while True:
        threshold = np.zeros(image.shape, np.uint8)
        trackbar_position_threshold = cv2.getTrackbarPos("threshold", window_name)
        trackbar_position_erode = cv2.getTrackbarPos("erode", window_name)
        trackbar_position_dilate = cv2.getTrackbarPos("dilate", window_name)
        color = cv2.getTrackbarPos("color", window_name)
        if color == 0:
            ret, threshold = cv2.threshold(image, trackbar_position_threshold, 255, cv2.THRESH_BINARY)
            #    ret, threshold = cv2.threshold(image, trackbar_position_threshold, 255, cv2.THRESH_BINARY_INV)
            #   threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY , 11, 2)
        else:
            threshold = image

        if trackbar_position_erode > 0:
            kernel = np.ones((trackbar_position_erode, trackbar_position_erode), np.uint8)
            threshold = cv2.erode(threshold, kernel)

        if trackbar_position_dilate > 0:
            kernel = np.ones((trackbar_position_dilate, trackbar_position_dilate), np.uint8)
            threshold = cv2.dilate(threshold, kernel)

        cv2.imshow(window_name, threshold)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return threshold


img_grey = cv2.imread('sample_anonyme.png', cv2.IMREAD_GRAYSCALE)
simple_trackbar(img_grey, "binary")
