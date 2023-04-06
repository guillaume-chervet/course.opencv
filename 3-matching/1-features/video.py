import cv2
import feature_matching
import image

#vid = cv2.VideoCapture(0)
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

template = cv2.imread("templates/book_deeplearning_800.jpg", cv2.IMREAD_GRAYSCALE)
template, ratio = image.normalize_size(template, 400)
match_func = feature_matching.apply_match(template, 15)

while (True):
    ret, frame = vid.read()
    try:
        img_matches = match_func(frame)
        if img_matches is not None:
            cv2.imshow('img_matches', img_matches)
    except Exception as e:
        print(e)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()