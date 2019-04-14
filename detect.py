import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import _function as f


video_capture = cv2.VideoCapture(0)

# infinite loop, break by key ESC
while True:
    if not video_capture.isOpened():
        print("cannot")
        break
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
        #minSize=(self.face_size, self.face_size)
    )
    # placeholder for cropped faces
    face_imgs = np.empty((len(faces)))
    for i, face in enumerate(faces):
        face_img, cropped = f.crop_face(frame, face, margin=50, size=112)
        (x, y, w, h) = cropped
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
    
    if len(face_imgs) > 0:
        tt = []
        tt.append(image.img_to_array(face_img))
        tt = np.array(tt, np.float32)
    
    # draw results
    for i, face in enumerate(faces):
        label = "face"
        f.draw_label(frame, (face[0], face[1]), label)
    cv2.imshow('Detect Faces', frame)
    if cv2.waitKey(5) == 27:  # ESC key press
        break