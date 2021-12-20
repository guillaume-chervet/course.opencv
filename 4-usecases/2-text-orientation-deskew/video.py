import time

import cv2

# define a video capture object
from text.text import is_image_contain_text
from text.deskew import deskew
from text.orientation import orientate


def rotate(angle, image):

    if angle == 0:
        return image

    if angle >= 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
        angle = angle - 180

    if angle <= -180:
        image = cv2.rotate(image, cv2.ROTATE_180)
        angle = angle + 180

    if 67.5 <= angle:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        angle = angle - 90

    if angle <= -67.5:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        angle = angle + 90

    if angle == 0:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img_cv = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_img_cv

vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    #is_img_with_text, distance, img_mser_cv = is_image_contain_text(frame)

    #if is_img_with_text:
    deskew_start = time.time()
    angle_deskew = deskew(frame)
    img_deskew_cv = rotate(angle_deskew, frame)
    deskew_duration = time.time() - deskew_start
    orientate_start = time.time()
    img_straighted_cv, angle_orientation, duration_orientation = orientate(img_deskew_cv)
    orientate_duration = time.time() - orientate_start

    value = [255,255,255]

    border = int(1280-720/2)

    img_straighted_cv = cv2.pyrDown(img_straighted_cv)
    row, col = img_straighted_cv.shape[:2]

    border_top = 0
    border_left=0
    if row > col:
        border_left = int(row-col)
    else:
        border_top = int(col-row)
    img_straighted_cv = cv2.copyMakeBorder(img_straighted_cv, 0, border_top, 0, border_left, cv2.BORDER_CONSTANT, None, value)
    cv2.imshow('frame', img_straighted_cv)
    cv2.imshow('orginal', cv2.pyrDown(frame))
   # else:
        # Display the resulting frame
    #    cv2.imshow('frame', frame)


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()