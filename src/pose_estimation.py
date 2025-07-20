import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseEstimator:
    """
    A class to estimate and draw human pose landmarks on video frames using MediaPipe Pose.

    Attributes:
        pose (mp.solutions.pose.Pose): MediaPipe Pose estimation model.
    """

    def __init__(self):
        """
        Initializes the PoseEstimator with MediaPipe Pose.

        Configuration:
        - Uses live video stream (not static image mode)
        - Medium model complexity
        - Disables segmentation
        - Sets minimum detection confidence to 0.5
        """
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def draw_pose(self, frame):
        """
        Detects and overlays pose landmarks on the given frame.

        Args:
            frame (np.ndarray): A BGR image/frame from video input.

        Returns:
            np.ndarray: The same BGR frame with pose landmarks drawn (if detected).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        return frame
