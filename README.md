# Car Counter

## Overview

This Python script uses computer vision techniques to count vehicles in a given video. The implementation is based on the YOLO (You Only Look Once) object detection model, integrated with a SORT (Simple Online and Realtime Tracking) algorithm for tracking.

## Features

- **Object Detection:** Utilizes the YOLO model to detect vehicles in the video.

- **Tracking:** Implements the SORT algorithm for real-time tracking of detected vehicles.

- **Counting:** Tracks and counts vehicles crossing predefined lines in the video.

- **Visualization:** Displays the output with graphical overlays indicating the total count and count in specific directions.


## Prerequisites

Make sure you have the following dependencies installed:

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)

- OpenCV (`cv2`)

- cvzone

- SORT (Simple Online and Realtime Tracking)

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rennycodes/car-counter.git
   cd car-counter
   ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Download the YOLO weights file `yolov8l.pt` and place it in the yolo weights directory.**


4. **Run the script:**
    ```bash
    python car_counter.py
    ```
## Configuration
Adjust the following parameters in the script as needed:  
- `lineUp` and `lineDown`: Define the lines for counting vehicles in the video.
- `mask`: Set the mask image for region of interest.

## Output
The processed video will be saved as `carcounter.mp4` in the same directory.

## Acknowledgments
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- SORT
