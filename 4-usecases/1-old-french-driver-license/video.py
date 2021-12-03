import cv2
from rectangle_detection import extract_rectangle

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    rectangle_img, homography_points = extract_rectangle(frame, "/", environment_mode="production")
    cv2.imshow('frame', cv2.pyrDown(rectangle_img))
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