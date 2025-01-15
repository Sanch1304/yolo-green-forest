import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img = cv2.imread('../resourses/traffic.jpg')
results = model(img)
results.print()
img = np.squeeze(results.render())#results.render() returns array of pixel array np.squeeze suueeze singal dimensional values
print(np.size(results.render()))
print('squeeze')
print(np.size(img))

# img = cv2.resize(img,(700,700))
cv2.imshow('image',img)
#
cv2.waitKey(0)
