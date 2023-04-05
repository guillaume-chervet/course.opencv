import pathlib
import cv2
import urllib.request

# Download the cascade (more available at https://github.com/opencv/opencv/tree/master/data/haarcascades)
model = 'haarcascade_frontalface_default.xml'
if pathlib.Path(model):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, model)

face_cascade = cv2.CascadeClassifier(model)
vid = cv2.VideoCapture(1)
while True:
    _, img = vid.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()