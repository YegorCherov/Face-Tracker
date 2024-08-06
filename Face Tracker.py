```python
import sqlite3
import cv2
import dlib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import os
import random


# Add a path to your video file here (e.g., "path/to/video.mp4")
VIDEO_FILE_PATH = "Walking.mp4"


# Database setup and functions
def create_database():
    conn = sqlite3.connect('face_recognition.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS faces
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL,
                     encoding BLOB NOT NULL)''')
    conn.close()

def add_face_encoding(encoding):
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    encoding_blob = encoding.tobytes()
    cursor.execute("INSERT INTO faces (name, encoding) VALUES ('Unknown', ?)", (encoding_blob,))
    face_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return face_id

def get_known_face_encodings():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, encoding FROM faces")
    known_faces = {}
    for row in cursor.fetchall():
        face_id, encoding_blob = row
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        known_faces[face_id] = encoding
    conn.close()
    return known_faces

# Camera stream class
class CameraStream:
    def __init__(self, source=0):
        # Try to initialize the camera. If it fails, use the video file.
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(VIDEO_FILE_PATH)

                # Fetch the video frame dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None

    def release(self):
        self.cap.release()

# GUI application class
class GUIApplication:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.camera = CameraStream()
        self.detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_recog = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        self.face_colors = {}
        self.canvas = tk.Canvas(window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()
        self.update_frame()

    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector(rgb_frame)
            known_faces = get_known_face_encodings()

            for detection in detections:
                face = detection.rect  # Get the standard rectangle from mmod_rectangle
                width = face.right() - face.left()
                height = face.bottom() - face.top()

                shape = self.predictor(rgb_frame, face)
                face_descriptor = np.array(self.face_recog.compute_face_descriptor(rgb_frame, shape))
                face_detected = False

                for face_id, known_encoding in known_faces.items():
                    matches = np.linalg.norm(face_descriptor - known_encoding) < 0.6
                    if matches:
                        face_detected = True
                        self.draw_face_rectangle(frame, face, face_id)
                        break

                if not face_detected:
                    new_face_id = add_face_encoding(face_descriptor)
                    known_faces[new_face_id] = face_descriptor
                    self.draw_face_rectangle(frame, face, new_face_id)
                    self.save_face_image(rgb_frame, face, new_face_id)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update_frame)

    def draw_face_rectangle(self, frame, face, face_id):
        if face_id not in self.face_colors:
            self.face_colors[face_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.face_colors[face_id], 2)