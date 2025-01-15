import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


class AttendanceSystem:
    def _init_(self, images_path, attendance_file):
        self.images_path = images_path
        self.attendance_file = attendance_file
        self.images = []
        self.classNames = []
        self.encodeFaceList = []

        self.load_images()
        self.encodeFaceList = self.find_encodings(self.images)
        print(f'Found {len(self.encodeFaceList)} encodings.')
        print('Encoding completed.')

    def load_images(self):
        myList = os.listdir(self.images_path)
        for cls in myList:
            curImg = cv2.imread(f'{self.images_path}/{cls}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cls)[0])
        print("Loaded image files:", myList)
        print("Class names:", self.classNames)

    def find_encodings(self, images):
        encodeList = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(image)[0]
            encodeList.append(encode)
        return encodeList

    def mark_attendance(self, name):
        with open(self.attendance_file, 'r+') as f:
            myDataList = f.readlines()
            nameList = [line.split(',')[0] for line in myDataList]

            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            imgS = cv2.resize(img, (0, 0), None, fx=0.25, fy=0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faceLocCurr = face_recognition.face_locations(imgS)
            encodeCurrFrame = face_recognition.face_encodings(imgS, faceLocCurr)

            for encodeFace, faceLoc in zip(encodeCurrFrame, faceLocCurr):
                matches = face_recognition.compare_faces(self.encodeFaceList, encodeFace)
                faceDist = face_recognition.face_distance(self.encodeFaceList, encodeFace)
                matchIndex = np.argmin(faceDist)

                if matches[matchIndex]:
                    name = self.classNames[matchIndex].upper()
                    print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), cv2.FILLED)
                    cv2.putText(img, f'Welcome: {name}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    self.mark_attendance(name)

            cv2.imshow("Webcam", img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to break
                break

        cap.release()
        cv2.destroyAllWindows()


# Usage example:
if _name_ == "_main_":
    images_path = "../Attendence"
    attendance_file = "../drivername.csv"
    attendance_system = AttendanceSystem(images_path, attendance_file)
    attendance_system.run()