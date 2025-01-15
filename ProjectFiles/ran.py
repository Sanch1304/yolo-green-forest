import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import uuid
import os
import time


Images_path = os.path.join('data2','images')
labels = ['awake','drowsy']
number_imgs = 20

os.makedirs(Images_path, exist_ok=True)

cap = cv2.VideoCapture(0)

for label in labels:
    print('collecting images for {}'.format(label))
    time.sleep(2)
    for img_num in range(number_imgs):
        print('collecting images for {},image num{}'.format(label,img_num))


        ret , frame = cap.read()
        print(ret)

        imgname = os.path.join(Images_path,label+'.'+str(uuid.uuid1()))
        cv2.imshow('image ',frame)
        cv2.imwrite(imgname,frame)

        for label in labels:
            print('collecting images for {}'.format(label))

        for img_num in range(number_imgs):
            print('collecting images for {},image num{}'.format(label,img_num))
            imgname = os.path.join(Images_path,label+'.'+str(uuid.uuid1())+'.jpg')
            print(imgname)

    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()