import os
from datetime import datetime, timedelta

# Define the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for storing employee photos
PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
# Directory for saving recorded violations
VIOLATIONS_DIR = os.path.join(BASE_DIR, "violations")
# Path to the serialized employee database file
DATABASE_FILE = os.path.join(BASE_DIR, "database", "employees.pkl")
# Path to the YOLO model weights
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# Confidence threshold for YOLO object detection
YOLO_CONF = 0.55
# Face similarity threshold (lower values mean stricter matching)
FACE_THRESHOLD = 0.35         
# Minimum time between logging the same person (prevents duplicate spam)
COOLDOWN_SECONDS = 45