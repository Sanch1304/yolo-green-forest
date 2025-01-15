import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import dlib
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
    with open('../attendence.csv','+r') as f:
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

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,fx=0.25,fy=0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faceLocCurr = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,faceLocCurr)
    # encode = face_recognition.face_encodings(img)[0]

    for encodeFace , FaceLoc in zip(encodeCurrFrame,faceLocCurr):
        matches = face_recognition.compare_faces(encodeFaceList,encodeFace)
        faceDist =face_recognition.face_distance(encodeFaceList,encodeFace)
        print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = FaceLoc
            y1,x2,y2,x1 = 4*y1,4*x2,4*y2,4*x1
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
            cv2.rectangle(img,(x1,y2-35),(x2,y2),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            markAtttendence(name)
    cv2.imshow("webcam",img)
    cv2.waitKey(5)