from sort.sort import Sort

class BallTracker:
    """
    A class to track multiple footballs across video frames using the SORT algorithm.

    Attributes:
        tracker (Sort): An instance of the SORT multi-object tracker.
    """

    def __init__(self):
        """
        Initializes the BallTracker with a configured SORT tracker.

        Configuration:
        - max_age: Number of frames to keep a track alive without detections (default: 30)
        - min_hits: Minimum detections before a track is considered valid (default: 3)
        - iou_threshold: Minimum IoU for association (default: 0.3)
        """
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    def update(self, detections):
        """
        Updates the tracker with the current frame's detections.

        Args:
            detections (np.ndarray): A NumPy array of shape (N, 5) containing detected bounding boxes.
                                     Each row represents [x1, y1, x2, y2, confidence].

        Returns:
            np.ndarray: A NumPy array of shape (M, 5) representing tracked objects with IDs.
                        Each row is [x1, y1, x2, y2, ID].
        """
        return self.tracker.update(detections)
