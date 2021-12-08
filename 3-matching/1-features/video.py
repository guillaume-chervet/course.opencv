
import cv2
import feature_matching
import image
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

template = cv2.imread("templates/img.png", 0)
template, ratio = image.normalize_size(template, 600)
match_func = feature_matching.apply_match(template)
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cv2.imshow('frame', frame)

    img_matches = match_func(frame)
    if img_matches is not None:
        cv2.imshow('img_matches', img_matches)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()