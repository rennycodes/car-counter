# Importation
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

# Variables
model = YOLO('../..yolo weights/yolov8l.pt')
vid = cv.VideoCapture('assets/cars.mp4')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

tracker = Sort(max_age = 22, min_hits = 3, iou_threshold = 0.3)
lineUp = [180, 410, 640, 410]
lineDown = [680, 400, 1280, 450]
countUp = []
countDown = []
totalCount = []
mask = cv.imread('assets/mask.png')

# Setting up video writer properties
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv.CAP_PROP_FPS)

# Writing the video writer
videoWriter = cv.VideoWriter(('carcounter.mp4'), cv.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

# Working on the video, model
while True:
    ref, frame = vid.read()
    frameRegion = cv.bitwise_and(frame, mask)
    result = model(frameRegion, stream = True)

    # Total count graphics
    frameGraphics = cv.imread('assets/graphics.png', cv.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, frameGraphics, (0,0))

    # Vehicle count graphics
    frameGraphics1 = cv.imread('assets/graphics1.png', cv.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, frameGraphics1, (420, 0))

    detections = np.empty((0, 5))

    # Working on the model result
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = (x2-x1), (y2-y1)


