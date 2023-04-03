import cv2
import argparse
import text_detection

#vid = cv2.VideoCapture(0)
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


ret, frame = vid.read()

parser = argparse.ArgumentParser(
    description="Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "
                "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)"
                "The OCR model can be obtained from converting the pretrained CRNN model to .onnx format from the github repository https://github.com/meijieru/crnn.pytorch"
                "Or you can download trained OCR model directly from https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing")
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', '-m', default="frozen_east_text_detection.pb",
                    help='Path to a binary .pb file contains trained detector network.')
parser.add_argument('--ocr', default="CRNN_VGG_BiLSTM_CTC.onnx",
                    help="Path to a binary .pb or .onnx file contains trained recognition network", )
parser.add_argument('--width', type=int, default=512,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=512,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--thr', type=float, default=0.5,
                    help='Confidence threshold.')
parser.add_argument('--nms', type=float, default=0.4,
                    help='Non-maximum suppression threshold.')
args = parser.parse_args()


text_detection.main(args)

