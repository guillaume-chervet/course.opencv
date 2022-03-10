"""
A fun project to make circles & rectangle 'magically' appear on the screen
Platform: Windows 10
Python Version: 3.10+
Major libraries: MediaPipe, OpenCV, NumPy
"""

import cv2
import numpy as np
import math

# Camera settings
DEFAULT_CAM = 0 # Built-in camera
USB_CAM = 1 # External camera connected via USB port

CAM_SELECTED = DEFAULT_CAM
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
FLIP_CAMERA_FRAME_HORIZONTALLY = True

# mediapipe parameters
MAX_HANDS = 2
DETECTION_CONF = 0.4
TRACKING_CONF = 0.5
MODEL_COMPLEX = 1
HAND_1 = 0
HAND_2 = 1
INDEX_FINGER_TIP = 8
X_COORD = 0
Y_COORD = 1

FIGURES_LIST = ["Circles", "MergedCircle", "Rectangle"]

# Drawing parameters - for opencv
CIR_RADIUS = 200
CIR_COLOR = (255, 0, 0)
CIR_THICKNESS = 3

MERG_CIR_COLOR = (0, 255, 0)
MERG_CIR_THICKNESS = 3

RECT_POINT = 300
RECT_COLOR = (0, 0, 255)
RECT_THICKNESS = 3



#https://github.com/jazir/MediaPiPe-Magic-Figures
class MpHands:
    import mediapipe as mp

    def __init__(self, max_hands=MAX_HANDS, det_conf=DETECTION_CONF, complexity=MODEL_COMPLEX, track_conf=TRACKING_CONF):
        """
        Inputs:-
        static_image_mode: Mode of input. If set to False, the solution treats the input images as a video stream.
        max_num_hands: Maximum number of hands to detect. Default to 2
        model_complexity: Complexity of the hand landmark model. 0 or 1.
                          Landmark accuracy as well as inference latency generally go up with the model complexity.
                          Default to 1.
        min_detection_confidence: Minimum confidence value ([0.0, 1.0]) from the hand detection model for the
                                  detection to be considered successful. Default to 0.5.
        min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the
                                 hand landmarks to be considered tracked successfully.
                                 Ignored if static_image_mode is True. Default to 0.5.
        Output:-
        multi_hand_landmarks: Collection of detected/tracked hands, where each hand is represented as a
                              list of 21 hand landmarks and each landmark is composed of x, y and z.
                              x and y are normalized to [0.0, 1.0] by the image width and height respectively.
        """
        self.hands = self.mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=max_hands,
                                                   model_complexity=complexity, min_detection_confidence=det_conf,
                                                   min_tracking_confidence=track_conf)

    def marks(self, video_frame):
        """
        Aim: To get the X & Y coordinates of all the 21 landmarks of both hands
        :param video_frame: captured frame from the opencv. This is in BGR format.
        :return my_hands: Array of hands with 21 landmarks (X & Y) of each hand
                          [[(h1_x0,h1_y0), (h1_x1,h1_y1), ...(h1_x20,h1_y20)],
                          [(h2_x0,h2_y0), (h2_x1,h2_y1), ...(h2_x20,h2_y20)], ...]
        """
        my_hands = []
        frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)  # opencv works in BGR, while rest of the world in RGB
        multi_hand_landmarks = self.hands.process(frame_rgb).multi_hand_landmarks
        if multi_hand_landmarks:    # Do the following if we have detected/tracked hands
            # multi_hand_landmarks is an array of arrays. Each array contains the 21 landmarks (in dict) of each hand
            for hand_landmarks in multi_hand_landmarks: # Stepping through each hand
                my_hand = []
                for land_mark in hand_landmarks.landmark:  # Stepping through the 21 landmarks of each hand
                    # landmark is a dict with x,y & z coordinates. We are interested in x & y only.
                    # Since x & y are normalized, multiply them with camera width and height to get the actual values.
                    # Finally, convert the coordinates into integers for opencv
                    my_hand.append((int(land_mark.x * CAM_WIDTH), int(land_mark.y * CAM_HEIGHT)))
                my_hands.append(my_hand)
        return my_hands


def calc_euclidean_dist(p1, p2):
    """
    Aim: Get the shortest distance between two points (Euclidean distance).
    :param p1: point 1 with (x1, y1) coordinates
    :param p2: point 2 with (x2, y2) coordinates
    :return euc_dist: Euclidean distance
    """
    (p1_x, p1_y) = p1
    (p2_x, p2_y) = p2
    euc_dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    return euc_dist


def select_figure(dist):
    """
    Aim: Select the figure to draw based on the distance
    :param dist: Distance between the index fingertips
    :return fig: selected figure
    """
    fig = FIGURES_LIST[0]
    if dist > CIR_RADIUS * 2:
        fig = FIGURES_LIST[0]
    if CIR_RADIUS * 2 >= dist > RECT_POINT:
        fig = FIGURES_LIST[1]
    if dist <= RECT_POINT:
        fig = FIGURES_LIST[2]
    return fig

# Camera configurations.
# Except 'CAM_SELECTED', all other settings are optional for faster launch of webcam in Windows
cam = cv2.VideoCapture(CAM_SELECTED, cv2.CAP_DSHOW) # CAP_DSHOW enables direct show without buffering
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)    # Set width of the frame
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)  # Set height of the frame
cam.set(cv2.CAP_PROP_FPS, CAM_FPS)  # Set fps of the camera
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Set the codec as 'MJPG'

findHands = MpHands()   # Create object

print("Press 'q' to quit")

while True:
    ignore, frame = cam.read()  # Read the frame from camera
    if FLIP_CAMERA_FRAME_HORIZONTALLY: # MediaPipe assumes the input image is mirrored. Flip it, if we want
        frame = cv2.flip(frame, 1)
    handData = findHands.marks(frame)   # Get the locations of both hands & fingers
    handDataLength = len(handData)  # Get the number of hands in the frame
    if handDataLength == 2: # We proceed only if there are two hands
        # The handData consists of 21 landmarks of each hand. We are interested in the tip of index fingers only.
        # Calculate the Euclidean distance between the index fingertips.
        indexTipsDist = calc_euclidean_dist(handData[HAND_1][INDEX_FINGER_TIP], handData[HAND_2][INDEX_FINGER_TIP])
        figure = select_figure(indexTipsDist)   # Based on the distance, select the figure to appear on screen

        if figure == "Circles":
                for hand in handData:   # Draw circles with index fingertips as centers
                    circleCenter = hand[INDEX_FINGER_TIP]
                    cv2.circle(frame, circleCenter, CIR_RADIUS, CIR_COLOR, CIR_THICKNESS)

        elif figure == "MergedCircle":
                # Draw a circle which encloses our fingertips at min level,
                # so that our fingertips will be on the edge of the circle
                point1 = handData[HAND_1][INDEX_FINGER_TIP]
                point2 = handData[HAND_2][INDEX_FINGER_TIP]
                (x, y), radius = cv2.minEnclosingCircle(np.array([point1, point2])) # points should be passed as a single numpy array
                mergedCircleCenter = (int(x), int(y))   # opencv wants integer values
                mergedCircleRadius = int(radius)
                cv2.circle(frame, mergedCircleCenter, mergedCircleRadius, MERG_CIR_COLOR, MERG_CIR_THICKNESS)

        elif figure == "Rectangle":   # Draw a rectangle with our index fingertips as diagonally opposite edges.
                point1 = (handData[HAND_1][INDEX_FINGER_TIP][X_COORD], handData[HAND_1][INDEX_FINGER_TIP][Y_COORD])
                point2 = (handData[HAND_2][INDEX_FINGER_TIP][X_COORD], handData[HAND_2][INDEX_FINGER_TIP][Y_COORD])
                cv2.rectangle(frame, point1, point2, RECT_COLOR, RECT_THICKNESS)
    else:
        print("Show both hands for the magic to happen or Press 'q' to quit")

    cv2.imshow('Magic Frame', frame)    # Display the frame
    cv2.moveWindow('Magic Frame', 0, 0) # Move the frame to the top-left corner of the monitor

    if cv2.waitKey(1) & 0xff == ord('q'):   # wait for letter 'q' to exit.
        print("Quiting program")
        break

cam.release()   # release the camera
cv2.destroyAllWindows() # close all frame windows