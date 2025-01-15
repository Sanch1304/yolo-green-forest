import cv2
import os
import time
import uuid
import torch
from matplotlib import pyplot as plt
import numpy as np

from ultralytics import YOLO

import os


file_path = 'C:/Users/asus/OneDrive/Documents/OneDrive/Desktop/yoloproject/yolov5/runs/train/exp11/weights/last.pt'

if os.path.exists(file_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=file_path, force_reload=True)
    print("Model loaded successfully")
else:
    print(f"File not found: {file_path}. Please check the path and try again.")

# imgpath = 'C:/Users/asus/OneDrive/Documents/OneDrive/Desktop/yoloproject/data3/images/awake.7a7a1d20-58d5-11ef-a45e-ae71b91e456e.jpg'
#
# img = cv2.imread(imgpath)
#
# results = model(img)
# print(results.render())
# cv2.imshow('yolo', np.squeeze(results.render()))







cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    results = model(frame)
    cv2.imshow('yolo', np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()