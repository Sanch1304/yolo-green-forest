import cv2
import os
import time
import uuid

# Define paths and parameters
Images_path = os.path.join('data4', 'images')
labels = ['awake', 'drowsy']
number_imgs = 60




# Initialize the camera
cap = cv2.VideoCapture(0)

for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(2)

    for img_num in range(number_imgs):
        print('Collecting images for {}, image num {}'.format(label, img_num))

        ret, frame = cap.read()

        if not ret:
            print('Failed to capture image')
            continue

        imgname = os.path.join(Images_path, label + '.' + str(uuid.uuid1()) + '.jpg')
        cv2.imshow('image', frame)
        cv2.imwrite(imgname, frame)
        time.sleep(1)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()