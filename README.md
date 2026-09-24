# SentinelVision

SentinelVision is a computer vision project I built to experiment with face recognition, person tracking and camera positioning.

The system uses **InsightFace** for face recognition and **YOLOv8** for person tracking. When a known face is detected, the program calculates its position relative to the center of the camera frame and gives a direction such as LEFT, RIGHT, UP, DOWN or CENTER.

The idea behind the project is to eventually connect these directions to a physical pan-tilt camera mechanism.

## Features

* Face detection using InsightFace
* Face recognition using face embeddings
* Local storage of face embeddings
* Cosine similarity for comparing faces
* YOLOv8 person tracking
* Face position calculation
* Horizontal and vertical direction detection
* Dead zone to reduce unnecessary movement
* Alternate-frame processing for better performance

## Technologies

* Python
* OpenCV
* NumPy
* InsightFace
* YOLOv8
* Ultralytics

## How it works

The basic workflow is:

```text
Known Face Images
       ↓
InsightFace
       ↓
Face Embeddings
       ↓
Save Embeddings
       ↓
Webcam
       ↓
Face Detection
       ↓
Face Recognition
       ↓
Find Face Position
       ↓
LEFT / RIGHT / UP / DOWN / CENTER
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/AliiUzair/SentinelVision.git
cd SentinelVision
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Add known faces

Create or use the existing folder:

```text
known faces
```

Add the images of the people you want to recognize.

For example:

```text
known faces/
├── Ali.jpg
├── Ahmed.jpg
└── Hamza.jpg
```

The filename is used as the person's name.

### 4. Run the program

```bash
python main.py
```

Press **Q** to close the webcam window.

## Face database

When the program finds a new image in the `known faces` folder, it generates a face embedding and saves it locally in:

```text
trained_faces.pkl
```

This file is ignored by Git and should not be uploaded to GitHub.

## Direction detection

The program compares the center of the detected face with the center of the camera frame.

For example:

```text
        UP
         ↑

LEFT ← CENTER → RIGHT

         ↓
       DOWN
```

A dead zone is used around the center so that small movements don't constantly change the direction.

## Current status

SentinelVision is still a work in progress.

The current version focuses mainly on the computer vision side of the project. The next step is to connect the direction output to a physical pan-tilt mechanism so that the camera can automatically follow the target.

## Future improvements

* Improve tracking when the face is temporarily lost
* Improve recognition under different lighting conditions
* Handle multiple people more reliably
* Improve YOLO tracking integration
* Add smoother camera movement
* Test the system with a physical pan-tilt setup

## Note

This project is mainly for learning and experimentation with computer vision and robotics.

Only use face images and recognition systems with the knowledge and permission of the people involved.
