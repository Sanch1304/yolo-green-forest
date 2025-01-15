import mediapipe as mp
import cv2
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

class_name = 'frustaded'

cap = cv2.VideoCapture(0)


with open('coordinatepose2.csv', mode='a', newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)


    landmarks = ['class']
    num_pose_landmarks = 33
    num_face_landmarks = 468
    # to cteate a list of values [x1,y1,z1,v1,x2,y2.......]
    # for val in range(1, num_pose_landmarks + num_face_landmarks + 1):
    #     landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    #
    # writer.writerow(landmarks)
#initializing holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=4, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=4, circle_radius=2))

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=4, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=4, circle_radius=2))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=4, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=4, circle_radius=2))

        image = cv2.resize(image, (980, 720))

        try:
            pose = results.pose_landmarks.landmark
            face = results.face_landmarks.landmark

            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row + face_row
            row.insert(0, class_name)


            with open('coordinatepose2.csv', mode='a', newline='') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(row)

        except:
            pass

        cv2.imshow('raw', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()