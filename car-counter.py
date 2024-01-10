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

            # Confidence
            conf = math.floor(box.conf[0]*100)/100

            # Classnames
            cls = int(box.cls[0])
            vehicleNames = classNames[cls]

            # Selecting the type of vehicle we want to detect
            if vehicleNames == 'car' or vehicleNames == 'bus' or vehicleNames == 'truck'\
                or vehicleNames == 'motorbike' and conf >= 0.3:
                currentDetection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentDetection))

    # Tracking codes
    trackerUpdate = tracker.update(detections)

    # Tracking lines
    cv.line(frame, (lineUp[0], lineUp[1]), (lineUp[2], lineUp[3]), (0, 0, 255), thickness = 3)
    cv.line(frame, (lineDown[0], lineDown[1]), (lineDown[2], lineDown[3]), (0, 0, 255), thickness = 3)

    for update in trackerUpdate:
        # Tracker bounding boxes and ID
        x1, y1, x2, y2, id = update
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = (x2-x1), (y2-y1)

        # Tracker circle
        cx, cy = (x1+w//2), (y1+h//2)
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        # Right side tracking code
        if lineUp[0] < cx < lineUp[2] and lineUp[1] - 5 < cy < lineUp[3] + 5:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv.line(frame, (lineUp[0], lineUp[1]), (lineUp[2], lineUp[3]), (0, 255, 0), thickness = 3)

            if countUp.count(id) == 0:
                countUp.append(id)

        # Left side tracking code
        if lineDown[0] < cx < lineDown[2] and lineDown[1] - 15 < cy < lineDown[3] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv.line(frame, (lineDown[0], lineDown[1]), (lineDown[2], lineDown[3]), (0, 255, 0), thickness = 3)

            if countDown.count(id) == 0:
                countDown.append(id)

        # Displaying ID and bounding boxes
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR = (255, 0, 255), rt = 1)
        cvzone.putTextRect(frame, f'{id}', (x1, y1), scale = 1, thickness = 2)

    # Displaying graphics texts(count)
    cv.putText(frame, str(len(totalCount)), (225, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
    cv.putText(frame, str(len(countUp)), (600, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
    cv.putText(frame, str(len(countDown)), (850, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)

    # Displaying the output result
    cv.imshow('vid', frame)

    # Writing out the frame
    videoWriter.write(frame)
    cv.waitKey(1)

# Closing down everything
vid.release()
cv.destroyAllWindows()
videoWriter.release()


