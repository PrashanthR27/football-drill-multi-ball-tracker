import cv2
import numpy as np

MOTION_HISTORY = 10
STATIONARY_THRESHOLD = 5.0

def draw_tracking_info(frame, tracks, action_ball_id=None, trails={}, stationary_ids=set()):
    for track in tracks:
        x1, y1, x2, y2, track_id = [int(v) for v in track]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if track_id == action_ball_id:
            label = f"ID{track_id} | Action Ball"
            color = (0, 0, 255)
            trails.setdefault(track_id, []).append((cx, cy))
            if len(trails[track_id]) > 30:
                trails[track_id] = trails[track_id][-30:]
            for i in range(1, len(trails[track_id])):
                cv2.line(frame, trails[track_id][i-1], trails[track_id][i], (0, 0, 255), 2)
        elif track_id in stationary_ids:
            label = f"ID{track_id} | Stationary Ball"
            color = (0, 255, 0)
        else:
            label = f"ID{track_id}"
            color = (255, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def classify_action_ball(id_locations):
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