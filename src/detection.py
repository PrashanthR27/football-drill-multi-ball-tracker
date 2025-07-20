from ultralytics import YOLO
import cv2

class BallDetector:
    def __init__(self, model_path='models\yolov8l.pt'):
        self.model = YOLO(model_path)

    def detect_balls(self, frame):
        results = self.model(frame)[0]
        balls = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            label = self.model.names[int(cls)]
            if label in ['sports ball', 'ball'] and conf > 0.3 and (x2 - x1) > 15 and (y2 - y1) > 15:
                balls.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
        return balls