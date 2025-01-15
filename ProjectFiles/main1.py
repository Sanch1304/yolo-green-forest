
from yolomodelz import YOLOModel

def main():
    file_path = 'C:/Users/asus/OneDrive/Documents/OneDrive/Desktop/yoloproject/yolov5/runs/train/exp8/weights/last.pt'
    yolo = YOLOModel(file_path)
    yolo.run_detection()

if __name__ == "__main__":
    main()