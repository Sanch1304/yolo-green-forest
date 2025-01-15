import cv2
import numpy as np
import face_recognition

cap = face_recognition.load_image_file('../resourses/Akanksha.jpeg')
cap = cv2.cvtColor(cap,cv2.COLOR_BGR2RGB)
testcap = face_recognition.load_image_file('../resourses/Akanksha Test.png')
testcap = cv2.cvtColor(testcap,cv2.COLOR_BGR2RGB)


faceloc = face_recognition.face_locations(cap)[0] #gives location of face in rectangular coordinate
encode = face_recognition.face_encodings(cap)[0]
cv2.rectangle(cap,(faceloc[1],faceloc[2]),(faceloc[3],faceloc[0]),(0,255,0),1)
faceloctest = face_recognition.face_locations(testcap)[0] #gives location of face in rectangular coordinate
encodetest = face_recognition.face_encodings(testcap)[0]
cv2.rectangle(testcap,(faceloctest[1],faceloctest[2]),(faceloctest[3],faceloctest[0]),(0,255,0),1)

results = face_recognition.compare_faces([encode],encodetest)
faceDis = face_recognition.face_distance([encode],encodetest)
a = results[0]
print(results)
print(faceDis)
if a:
    cv2.putText(cap,f'{results}',(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
    print('2')
else:
    cv2.putText(cap,f'{results}',(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
    print('1')
cv2.imshow('image',cap)


cv2.imshow('image2',testcap)
cv2.waitKey(0)