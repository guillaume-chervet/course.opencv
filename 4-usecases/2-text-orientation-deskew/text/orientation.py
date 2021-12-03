import os

import cv2
import numpy as np
import time



def find_orientation(image, template):
    img_rgb = image

    if len(img_rgb.shape) == 3:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    #w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where(res >= threshold)
    count = 0
    for pt in zip(*loc[::-1]):
        count=count+1
       # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    #print("count: " + str(count))
   # cv2.imshow('Detected Point', img_rgb)
   # cv2.waitKey()


    return count


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
template = cv2.imread(os.path.join(BASE_PATH, "query.png"), 0)
def orientate(image):
    start_time = time.time()
    images = [image]
    counts = []
    counts.append(find_orientation(image, template))
    image = cv2.rotate(image, cv2.ROTATE_180)
    images.append(image)
    counts.append(find_orientation(image, template))

    max_count = 0
    max_index = 0
    for index, count in enumerate(counts):
        #print(str(count))
        #print(str(index))

        if count > max_count:
            max_count = count
            max_index = index

    duration = time.time() - start_time
    #print("orientate time : " + str(straightening_duration))

    return images[max_index], (max_index*180), duration

