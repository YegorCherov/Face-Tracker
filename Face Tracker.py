import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class FaceDetector:
    def __init__(self):
        self.frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)
        profile_faces = self.profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)

        # Properly handle cases where faces or profile_faces are empty
        if len(faces) == 0 and len(profile_faces) == 0:
            return np.array([])  # No faces detected
        elif len(faces) == 0:
            return profile_faces
        elif len(profile_faces) == 0:
            return faces
        else:
            return np.concatenate((faces, profile_faces))

    
class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None

    def release(self):
        self.cap.release()

class GUIApplication:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.camera = CameraStream()
        self.detector = FaceDetector()

        self.canvas = tk.Canvas(window, width=self.camera.cap.get(3), height=self.camera.cap.get(4))
        self.canvas.pack()

        self.update_frame()

    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is not None:
            faces = self.detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update_frame)

    def run(self):
        self.window.mainloop()

    def on_closing(self):
        self.camera.release()
        self.window.destroy()

# Running the GUI Application
if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApplication(root, "Face Detection GUI")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.run()
