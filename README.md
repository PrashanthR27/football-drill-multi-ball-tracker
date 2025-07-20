# Football Drill Multi-Ball Tracker

A computer vision project to detect and track multiple footballs (stationary and action) in football drill videos using YOLOv8 for object detection and SORT for multi-object tracking. The goal is to dynamically overlay bounding boxes and ID-consistent tracking visuals on video frames.

---

## 🚀 Features

* Real-time football detection using YOLOv8
* Unique ID tracking of multiple footballs using SORT
* Classification of action ball (moving) vs stationary balls
* Trail overlay for action ball visualization
* Robust ID consistency across occlusion and overlaps

---

## 📁 Folder Structure

```
football-drill-multi-ball-tracker/
├── data/
│   └── raw_videos/            # Input test1.mp4, test2.mp4
├── outputs/   # Output annotated videos
├── models/    # yolov8l.pt
├── src/
│   ├── detection.py           # YOLO detection module
│   ├── tracking.py            # SORT wrapper
│   ├── utils.py               # Helper functions
│   ├── main.py                # Main execution script
│   └── sort/                  # SORT tracker code (from abewley repo)
├── report/
│   └── report.pdf             # Methodology report
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
```

---

## 🧰 Installation

### 1. Clone the repository

```bash
git clone https://github.com/PrashanthR27/football-drill-multi-ball-tracker.git
cd football-drill-multi-ball-tracker
```

### 2. Set up a Python virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 model weights (first-time only)

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Automatically downloads model and save it in folder models/
```

---

## ▶️ How to Run

### For test1.mp4:

```bash
python src/main.py
```

Ensure `video_name = "test1"` is set in `main.py`

### For test2.mp4:

Update this line in `main.py`:

```python
video_name = "test2"
```

Then run:

```bash
python src/main.py
```

Output video will be saved to:

```
outputs/test1_output.avi
outputs/test2_output.avi
```

---

## 📌 Notes

* The action ball is determined using motion tracking across frames.
* Stationary balls are identified based on minimal displacement.
* Trail visualization improves interpretability of movement.

---

## 📜 Credits

* YOLOv8 by Ultralytics
* SORT tracker by abewley (MIT License)

---

