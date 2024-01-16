# Car Counting with YOLO and SORT

This Python script utilizes the YOLO (You Only Look Once) model and SORT (Simple Online and Realtime Tracking) algorithm to count the number of cars in a given video. The code uses the Ultralytics YOLO implementation, OpenCV, and other libraries for efficient object detection and tracking.

## Requirements
Make sure to install the required libraries using the following:

```bash
pip install ultralytics opencv-python opencv-python-headless numpy

## Setup

Yolo Weights: Download the YOLO weights file (e.g., yolov8l.pt) and place it in the yolo weights directory.

Video File: Provide the video file containing the traffic footage (e.g., cars.mp4). Update the vid = cv.VideoCapture('assets/cars.mp4') line with the correct path if necessary.

Graphics Assets: Ensure that the assets folder contains the necessary graphics files (graphics.png and graphics1.png) for displaying count information.

Mask Image: Provide a mask image (mask.png) for region-of-interest extraction. This is used to focus on the relevant part of the video.
