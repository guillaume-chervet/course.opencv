import numpy as np
import cv2


def centroid_histogram(labels):
    numLabels = np.arange(0, len(np.unique(labels)) + 1)
    (hist, _) = np.histogram(labels, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    return bar


def kmean(image):
    NCLUSTERS = 7
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

    hist = centroid_histogram(labels)
    bar = plot_colors(hist, centers)
    return bar


img = cv2.imread('sample_anonyme.png', cv2.IMREAD_COLOR)
img_out = kmean(img)
cv2.imshow('img_out', img_out)
cv2.waitKey()
cv2.destroyAllWindows()
