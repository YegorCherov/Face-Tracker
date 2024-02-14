
# Face Detection & Tracking with Database Integration

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)

An advanced project that not only detects and tracks faces but also dynamically stores and retrieves face data using a SQLite database. This enhanced version is capable of handling multi-target face detection and tracking, including side faces, with the added functionality of face data management.

## Introduction

This project extends the capabilities of standard face detection by integrating a database to store face encodings, providing a more robust solution for applications requiring face recognition and tracking.

* **Dependencies**:
  * Python 3.10+
  * Dlib
  * Numpy
  * OpenCV-python
  * SQLite3
  * Tkinter
  * PIL (Python Imaging Library)

## Features

* **Face Detection and Tracking**: Utilizes Dlib's CNN face detection model for accurate face detection and tracking.
* **Dynamic Face Data Management**: Stores face encodings in a SQLite database for efficient retrieval and management.
* **Unique Face Identification**: Assigns a unique ID and color to each detected face, displayed in real-time on the GUI.
* **Fallback to Video File**: In the absence of a camera feed, the application can process a pre-recorded video.
* **Automatic Face Extraction**: Captures and saves images of detected faces in a dedicated folder for further use.
* **GUI Interface**: Features a Tkinter-based graphical user interface for real-time face tracking display.

## Setup and Run

* To run the application, execute the following command in your terminal:

  ```sh
  python3 FaceTracker.py
  ```

* The application will use the default camera. If unavailable, it will fallback to processing the video file specified in `VIDEO_FILE_PATH`.

* Detected faces and their encodings are stored in a SQLite database (`face_recognition.db`). Images of newly detected faces are saved in the `new_faces` folder.

## Application Use-Cases

* Ideal for developing training sets for facial recognition systems.
* Suitable for real-time face tracking applications in security, marketing, and user interaction domains.
* Backend integration for advanced face recognition and data analysis.

## Results
![alt text](https://github.com/Rozcy/Face-Tracker/blob/main/Gifs/FaceTrackerExampleCropped.gif)


