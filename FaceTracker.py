import cv2
import numpy as np
import tensorflow as tf
import threading
import queue
import tkinter as tk
from PIL import Image, ImageTk

class FaceDetector:
    def __init__(self):
        # Load the SSD MobileNet V2 FPNLite model
        self.model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model')

    def detect_faces(self, frame):
        # Convert the frame to RGB and resize to 640x640
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        # Prepare the frame for the model (expand dimensions)
        input_tensor = tf.convert_to_tensor(np.expand_dims(frame_resized, 0), dtype=tf.float32)
        # Run detection
        detections = self.model(input_tensor)
        # Process detections
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        # Filter out detections with low confidence
        faces = [(box, score) for box, score in zip(boxes, scores) if score > 0.5]
        return faces

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.frame_queue = queue.Queue(maxsize=10)
        self.read_thread = threading.Thread(target=self.update, args=())
        self.read_thread.daemon = True
        self.stopped = False

    def start(self):
        self.read_thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    break
                self.frame_queue.put(frame)

    def read(self):
        return self.frame_queue.get()

    def stop(self):
        self.stopped = True
        self.read_thread.join()
        self.cap.release()

def draw_faces(frame, faces):
    H, W = frame.shape[:2]
    for box, _ in faces:
        # Scale the bounding box coordinates back to the frame size
        (startY, startX, endY, endX) = box
        startX, startY, endX, endY = int(startX * W), int(startY * H), int(endX * W), int(endY * H)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return frame

class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = VideoStream(self.video_source).start()
        self.face_detector = FaceDetector()

        self.canvas = tk.Canvas(window, width = 640, height = 480)
        self.canvas.pack()

        self.update()

        self.window.mainloop()

def update(self):
    frame = self.vid.read()
    if frame is not None:
        print("Frame captured")
        faces = self.face_detector.detect_faces(frame)
        if faces:
            print(f"{len(faces)} face(s) detected")
        frame = draw_faces(frame, faces)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    else:
        print("No frame captured")
    self.window.after(15, self.update)


# Create a window and pass it to the Application object
Application(tk.Tk(), "Real-Time Face Detection")
