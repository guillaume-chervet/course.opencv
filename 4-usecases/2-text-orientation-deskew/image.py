import cv2


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