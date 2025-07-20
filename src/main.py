import cv2
from detection import BallDetector
from tracking import BallTracker
from utils import draw_tracking_info, classify_action_ball, find_stationary_ids
from collections import defaultdict
import os
import numpy as np

video_name = "test"
input_path = f"data/raw_videos/{video_name}.mp4"
output_path = f"outputs/{video_name}_output.avi"

os.makedirs("outputs", exist_ok=True)

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {input_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30

writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
if not writer.isOpened():
    raise IOError(f"Failed to open video writer for: {output_path}")

ball_detector = BallDetector()
ball_tracker = BallTracker()
id_locations = defaultdict(list)
trails = {}

frame_count = 0
stationary_ids = set()
action_ball_id = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or can't read the frame.")
        break

    detections = ball_detector.detect_balls(frame)
    detections_array = [det[:5] for det in detections]
    if len(detections_array) == 0:
        input_array = np.empty((0, 5), dtype=np.float32)
    else:
        input_array = np.array(detections_array, dtype=np.float32)

    tracks = ball_tracker.update(input_array)

    for x1, y1, x2, y2, track_id in tracks:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        id_locations[int(track_id)].append((cx, cy))

    if frame_count > 30:
        action_ball_id = classify_action_ball(id_locations)
        stationary_ids = find_stationary_ids(id_locations, action_ball_id)

    frame = draw_tracking_info(frame, tracks, action_ball_id, trails, stationary_ids)

    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    writer.write(frame)
    frame_count += 1

cap.release()
writer.release()
print(f"Video saved to {output_path}")
