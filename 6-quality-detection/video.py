
import cv2

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
obj = cv2.quality.QualityBRISQUE_create("brisque_model_live.yml",
                                               "brisque_range_live.yml")

while (True):

    ret, frame = vid.read()

    score = obj.compute(frame)
    print(score)

    cv2.imshow('orginal', cv2.pyrDown(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


