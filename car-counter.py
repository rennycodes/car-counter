# Importation
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

# Variables
model = YOLO('../..yolo weights/yolov8l.pt')
vid = cv.VideoCapture('assets/cars.mp4')
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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
line_up = [180, 410, 640, 410]
line_down = [680, 400, 1280, 450]
count_up = []
count_down = []
total_count = []
mask = cv.imread('assets/mask.png')

