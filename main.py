import cv2
import pickle
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from config import *

# Load YOLO model for helmet (0) and head (1) detection
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Load database of employee face embeddings
print("Loading face database...")
with open(DATABASE_FILE, "rb") as f:
    employees = pickle.load(f)

# Initialize InsightFace for face recognition
print("Loading InsightFace...")
face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Anti-spam: last alert time per recognized name
last_alert = {}

# Track each person by track_id
tracked = defaultdict(lambda: {
    'start_no_helmet': None,      # time when first seen without helmet
    'duration': 0.0,              # accumulated time without helmet
    'name': "Unknown"             # recognized name
})

print("Starting camera... (press Q to exit)")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

APPROX_FPS = 15.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera read error")
        break

    now = datetime.now()

    # Run detection + tracking
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=YOLO_CONF,
        verbose=False
    )

    annotated = results[0].plot()

    current_ids = set()

    # First pass: check which track_ids have helmet (class 0)
    has_helmet = defaultdict(bool)
    for box in results[0].boxes:
        if box.id is None:
            continue
        tid = int(box.id)
        current_ids.add(tid)
        if int(box.cls) == 0:  # helmet
            has_helmet[tid] = True

    # Process each detection
    for box in results[0].boxes:
        if box.id is None:
            continue

        tid = int(box.id)
        cls = int(box.cls)

        person = tracked[tid]

        if cls == 1:  # head = no helmet
            # Run face recognition every time we see a head
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]
            faces = face_analyzer.get(face_crop)

            if faces:
                emb = faces[0].normed_embedding
                min_dist = 999
                best_name = "Unknown"
                for name, emp_emb in employees.items():
                    dist = np.linalg.norm(emp_emb - emb)
                    if dist < min_dist:
                        min_dist = dist
                        best_name = name
                if min_dist <= FACE_THRESHOLD:
                    person['name'] = best_name

            # Count time only if no helmet detected for this track_id in this frame
            if not has_helmet[tid]:
                if person['start_no_helmet'] is None:
                    person['start_no_helmet'] = now
                person['duration'] += 1.0 / APPROX_FPS

                # Violation alert with cooldown
                name = person['name']
                if name not in last_alert or (now - last_alert[name]) >= timedelta(seconds=COOLDOWN_SECONDS):
                    ts = now.strftime("%Y-%m-%d_%H-%M-%S")
                    fn = f"{name}_{ts}.jpg"
                    path = os.path.join(VIOLATIONS_DIR, fn)
                    cv2.imwrite(path, annotated)

                    print(f"VIOLATION: {name} (ID {tid}) without helmet {person['duration']:.1f}s")
                    print(f"Photo saved: {fn}")

                    last_alert[name] = now

    # Reset timer if helmet is present
    for tid in current_ids:
        if has_helmet[tid]:
            person = tracked[tid]
            if person['start_no_helmet'] is not None:
                print(f"ID {tid} | Helmet put on | Was without: {person['duration']:.1f}s")
                person['start_no_helmet'] = None
                person['duration'] = 0.0

    # Remove old tracks
    for tid in list(tracked):
        if tid not in current_ids:
            del tracked[tid]

    cv2.imshow("Helmet Safety", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera stopped.")