from ultralytics import YOLO
import cv2

class BallDetector:
    """
    A class to detect footballs in video frames using a YOLOv8 model.

    Attributes:
        model (YOLO): An instance of the YOLO object detection model.
    """

    def __init__(self, model_path='models\yolov8l.pt'):
        """
        Initialize the BallDetector with a YOLO model.

        Args:
            model_path (str): Path to the YOLOv8 model weights file.
                              Default is 'models\\yolov8l.pt'.
        """
        self.model = YOLO(model_path)

    def detect_balls(self, frame):
        """
        Detects footballs in the given frame using the YOLO model.

        Args:
            frame (np.ndarray): A single image/frame (BGR format) from a video.

        Returns:
            List[List[float]]: A list of detected ball bounding boxes.
                               Each box is represented as [x1, y1, x2, y2, confidence].
        """
        results = self.model(frame)[0]
        balls = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            label = self.model.names[int(cls)]
            if label in ['sports ball', 'ball'] and conf > 0.3 and (x2 - x1) > 15 and (y2 - y1) > 15:
                balls.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
        return balls