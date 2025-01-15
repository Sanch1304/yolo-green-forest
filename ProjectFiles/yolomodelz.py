# yolo_model.py
import cv2
import os
import torch
import numpy as np

class YOLOModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = self.load_model()

    def load_model(self):
        if os.path.exists(self.file_path):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.file_path, force_reload=False)
            print("Model loaded successfully")
            return model
        else:
            raise FileNotFoundError(f"File not found: {self.file_path}. Please check the path and try again.")

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()



            results = self.model(frame)
            cv2.imshow('yolo', np.squeeze(results.render()))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()