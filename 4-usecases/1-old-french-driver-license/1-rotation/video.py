import cv2
from rectangle_detection import extract_rectangle

#vid = cv2.VideoCapture(0)
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while (True):

    ret, frame = vid.read()

    rectangle_img, homography_points = extract_rectangle(frame, "/", environment_mode="production")
    cv2.imshow('frame', cv2.pyrDown(rectangle_img))
    cv2.imshow('orginal', cv2.pyrDown(frame))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()