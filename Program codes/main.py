import cv2
import dlib
from imutils import face_utils
import numpy as np
from cvzone.SerialModule import SerialObject
from playsound import playsound
import torch
import face_recognition
import os
from datetime import datetime
from recentdriver import LastDriver
import pickle
import mediapipe as mp
import pandas as pd



mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_counter =0


with open('driver_language1.pkl','rb') as f :
    model1 = pickle.load(f)




class YOLOModel:
    def _init_(self, file_path):
        self.file_path = file_path
        self.model = self.load_model()

    def load_model(self):
        if os.path.exists(self.file_path):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.file_path, force_reload=False)
            print("Model loaded successfully")
            return model
        else:
            raise FileNotFoundError(f"File not found: {self.file_path}. Please check the path and try again.")

    def run_detection(self, frame):
        results = self.model(frame)
        return np.squeeze(results.render())

path = "../Attendence"
images = []
classNames =[]
myList = os.listdir(path)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(myList)
# print(images)
print(classNames)

def findEncoding(images):
    encodeList =[]
    for image in images:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList

def markAtttendence(name):
    with open('../drivername.csv', '+r') as f:
        myDatalist = f.readlines()
        namelist =[]
        print(myDatalist)
        for line in myDatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')




encodeFaceList = findEncoding(images)
print(len(encodeFaceList))
print('encoding completed')





# Initialize video capture
cap = cv2.VideoCapture(0)
arduino = SerialObject("COM5")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../Model/shape_predictor_68_face_landmarks.dat")

# Initialize YOLO model
file_path = 'C:/Users/asus/OneDrive/Documents/OneDrive/Desktop/yoloproject/yolov5/runs/train/exp8/weights/last.pt'
yolo = YOLOModel(file_path)

sleep = 0
drowsy = 0
active = 0
status = 0
color = (0, 0, 0)

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frametrain = frame.copy()
        imagetrain = cv2.cvtColor(frametrain, cv2.COLOR_BGR2RGB)
        results = holistic.process(imagetrain)
        imagetrain = cv2.cvtColor(imagetrain, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        face_frame = frame.copy()
        y_frame = frame.copy()

        # costum trained module using poseEstimation
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                imagetrain, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                imagetrain, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=4, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=4, circle_radius=2))

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                imagetrain, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=4, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=4, circle_radius=2))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                imagetrain, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=4, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=4, circle_radius=2))

        imagetrain = cv2.resize(imagetrain, (720, 480))
        try:
            pose = results.pose_landmarks.landmark
            face = results.face_landmarks.landmark

            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row + face_row




            X = pd.DataFrame([row])
            body_language_class = model1.predict(X)[0]
            body_language_prob = model1.predict(X)[0]
            print(body_language_class,body_language_prob)
            if body_language_class == 'driving':
                mp_counter +=1
                print(mp_counter)


        except:
            pass







        # for 68 facial landmarks
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink != 0 or right_blink != 0:
                a = 0
            elif left_blink == 1 or right_blink == 1:
                a = 1
            else:
                a = 2

            print(f"Left blink: {left_blink}, Right blink: {right_blink}, a: {a}")  # Add this line to print blink values

            if a == 2:
                drowsy = 0
                active = 0
                sleep += 1
                print(f"Sleep counter: {sleep}")  # Add this line to print sleep counter
                if sleep > 12:
                    cv2.putText(face_frame, "sleeping!!!", (100, 100), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255))
                    arduino.sendData([1])
                    playsound(r'../assets/buzzer2.wav')


            elif a == 1:
                sleep = 0
                active = 0
                drowsy += 1
                print(f"Drowsy counter: {drowsy}")  # Add this line to print drowsy counter
                if drowsy > 40:
                    cv2.putText(face_frame, "drowsy", (100, 100), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255))
                    arduino.sendData([0])

            elif a == 0:
                active += 1
                drowsy = 0
                sleep = 0
                print(f"Active counter: {active}")  # Add this line to print active counter
                if active > 40:
                    cv2.putText(face_frame, "active", (100, 100), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0))
                    arduino.sendData([0])


            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 0, 0), -1)



        yolo_frame = yolo.run_detection(y_frame)

        cv2.imshow("yolo", yolo_frame)
        cv2.imshow("result", face_frame)
        cv2.imshow("original", frame)
        cv2.imshow('raw', imagetrain)
        if mp_counter==40:
            print('worked')
            playsound('../assets/buzzer3.wav')
            mp_counter = 0
        key = cv2.waitKey(2)
        if key == 27:
            break
if _name_ == "_main_":
    images_path = "../Attendence"
    driver_list = "../drivername.csv"
    last_driver = LastDriver(images_path, driver_list)
    last_driver.run()


print('bye bye !!')