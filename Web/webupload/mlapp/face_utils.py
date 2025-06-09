import cv2
import dlib
import numpy as np

# Load face detector dan landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("mlapp/ml_model/shape_predictor_68_face_landmarks.dat")

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    return frame[y:y+h, x:x+w]

def extract_mouth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    coords = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
    x, y, w, h = cv2.boundingRect(coords)
    return frame[y:y+h, x:x+w]
