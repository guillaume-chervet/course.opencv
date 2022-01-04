import pathlib
import urllib.request
import numpy as np
import cv2

def kmean(image):
    NCLUSTERS = 5
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

image = 'guile.png'
if pathlib.Path(image):
    url = "https://th.bing.com/th/id/OIP.OGMAb0rahKATlVnmO0ZZMwHaKH?pid=ImgDet&rs=1"
    urllib.request.urlretrieve(url, image)


img = cv2.imread(image, cv2.IMREAD_COLOR)
img_out = kmean(img)
cv2.imshow('img_out', img_out)
cv2.waitKey()
cv2.destroyAllWindows()
