import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import uuid
import os
import time


cap = cv2.VideoCapture(0)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

while cap.isOpened():
    success, img = cap.read()
    results = model(img)
    img = np.squeeze(results.render())
    cv2.imshow('image',img)

    cv2.waitKey(10)