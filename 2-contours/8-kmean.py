import numpy as np
import cv2

def kmean(image):
    NCLUSTERS = 4
    NROUNDS = 1

    height, width, channels = image.shape
    samples = np.zeros([height * width, 3], dtype=np.float32)
    count = 0

    for x in range(height):
        for y in range(width):
            samples[count] = image[x][y]  # BGR color
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
                                              NCLUSTERS,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                              NROUNDS,
                                              cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image.shape)

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while (True):

    ret, frame = vid.read()

    img_out = kmean(frame)
    cv2.imshow('img_out', img_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
