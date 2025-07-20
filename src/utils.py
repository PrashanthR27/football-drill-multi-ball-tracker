import cv2
import numpy as np

# Constants
MOTION_HISTORY = 10  # Number of recent frames to consider for motion analysis
STATIONARY_THRESHOLD = 5.0  # Max movement distance to classify a ball as stationary

def draw_tracking_info(frame, tracks, action_ball_id=None, trails={}, stationary_ids=set()):
    """
    Draws tracking information on the frame for all tracked objects.

    Args:
        frame (np.ndarray): The current video frame (BGR format).
        tracks (List[List[int]]): A list of tracks where each track is [x1, y1, x2, y2, ID].
        action_ball_id (int): The ID of the action ball, which should have a trail.
        trails (dict): Dictionary storing the movement trail per ID.
        stationary_ids (set): Set of IDs identified as stationary balls.

    Returns:
        np.ndarray: The annotated frame with tracking boxes and labels.
    """
    for track in tracks:
        x1, y1, x2, y2, track_id = [int(v) for v in track]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if track_id == action_ball_id:
            label = f"ID{track_id} | Action Ball"
            color = (0, 0, 255)  # Red
            trails.setdefault(track_id, []).append((cx, cy))
            if len(trails[track_id]) > 30:
                trails[track_id] = trails[track_id][-30:]
            for i in range(1, len(trails[track_id])):
                cv2.line(frame, trails[track_id][i-1], trails[track_id][i], (0, 0, 255), 2)
        elif track_id in stationary_ids:
            label = f"ID{track_id} | Stationary Ball"
            color = (0, 255, 0)  # Green
        else:
            label = f"ID{track_id}"
            color = (255, 255, 255)  # White

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def classify_action_ball(id_locations):
    """
    Identifies the action ball as the ball with the highest recent motion.

    Args:
        id_locations (dict): Dictionary mapping track ID to list of (x, y) positions across frames.

    Returns:
        int or None: ID of the action ball, or None if no candidates available.
    """
    motions = {}
    for track_id, points in id_locations.items():
        if len(points) < 2:
            continue
        recent = points[-MOTION_HISTORY:] if len(points) > MOTION_HISTORY else points
        motions[track_id] = sum(
            np.linalg.norm(np.array(recent[i]) - np.array(recent[i - 1]))
            for i in range(1, len(recent))
        )
    return max(motions, key=motions.get) if motions else None

def find_stationary_ids(id_locations, action_ball_id):
    """
    Identifies stationary balls based on minimal movement across recent frames.

    Args:
        id_locations (dict): Dictionary mapping track ID to list of (x, y) positions across frames.
        action_ball_id (int): The ID of the action ball to exclude from stationary check.

    Returns:
        set: Set of track IDs considered to be stationary.
    """
    stationary_ids = set()
    for track_id, points in id_locations.items():
        if track_id == action_ball_id or len(points) < 2:
            continue
        recent = points[-MOTION_HISTORY:] if len(points) > MOTION_HISTORY else points
        motion = sum(
            np.linalg.norm(np.array(recent[i]) - np.array(recent[i - 1]))
            for i in range(1, len(recent))
        )
        if motion < STATIONARY_THRESHOLD:
            stationary_ids.add(track_id)
    return stationary_ids
