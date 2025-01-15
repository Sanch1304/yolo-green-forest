import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO


cap = cv2.VideoCapture(0)
model = YOLO('../Yolo_weights/yolov5lu.pt')

cap.set(3,480)
cap.set(4,720)
while True:
    success ,img = cap.read()
    results = model(img,show=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            print(x1,y1,x2,y2)
    # cv2.imshow('image',img)
    cv2.waitKey(1)
