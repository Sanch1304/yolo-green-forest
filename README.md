Driver Drowsiness Detection using MediaPipe, Random Forest, and YOLO Model
Overview
This repository contains a project for detecting driver drowsiness using a combination of MediaPipe for face and landmark detection, Random Forest for classification, and YOLO (You Only Look Once) for object detection. The system monitors the driver’s state, providing warnings for signs of drowsiness based on facial features and behaviors.

Key Components:
MediaPipe: Used for detecting facial landmarks and tracking the face for drowsiness signals like eye blink rates.
Random Forest: A classification algorithm to predict drowsiness state based on the extracted features from the driver’s face.
YOLO: For detecting surrounding objects and estimating the driver’s attention to the road.
Requirements
Python 3.x
MediaPipe
OpenCV
scikit-learn
YOLO (pre-trained weights)
Installation
To set up the project locally, follow these steps:

Clone this repository:
bash
Copy code
git clone https://github.com/Sanch1304/yolo-green-forest.git
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Data
The data folder includes:

train_data: Data used to train the models. It includes labeled images of drivers with different drowsiness states.
test_data: Data used to evaluate the trained models.
Training the Model
To train the Random Forest classifier and the YOLO object detection model, run:

bash
Copy code
python src/train_model.py
This script trains the Random Forest on the facial features extracted by MediaPipe and fine-tunes the YOLO model for object detection.

Execution
To run the driver drowsiness detection system on video input, use the following command:

bash
Copy code
python src/execute_model.py --video_path path/to/your/video
This script will load the trained models, detect the driver’s face, estimate drowsiness, and provide real-time alerts.

Contributions
Feel free to fork this repository and contribute by submitting pull requests. All contributions are welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

Make sure to replace your-username with your actual GitHub username. Let me know if you'd like to adjust or add anything specific!
