
import cv2

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
obj = cv2.quality.QualityBRISQUE_create("brisque_model_live.yml",
                                               "brisque_range_live.yml")
while (True):

    ret, frame = vid.read()

    # read image
   # img = cv2.imread(frame, 1)  # mention img_path
    # compute brisque quality score via static method
    #score = cv2.quality.QualityBRISQUE_compute(frame, "brisque_model_live.yml",
    #                                           "brisque_range_live.yml")  # specify model_path and range_path
    # compute brisque quality score via instance
    # specify model_path and range_path
   #
    score = obj.compute(frame)
    #cv2.imshow('mser', img_mser)
    cv2.imshow('orginal', cv2.pyrDown(frame))
    print(score)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
