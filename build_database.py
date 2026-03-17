# build_database.py
# This script creates a database of employee face embeddings from photos in the 'photos' folder.

import os
import cv2
import pickle
from insightface.app import FaceAnalysis

# Adjust these paths if your folder structure is different
PHOTOS_DIR = "photos"                  # Folder with employee photos (filename = name.jpg)
DATABASE_DIR = "database"              # Where to save the pickle file
DATABASE_FILE = os.path.join(DATABASE_DIR, "employees.pkl")

# Load the face analysis model (first run may take time to download models)
print("Loading face recognition model... (may take a minute on first run)")
face_analyzer = FaceAnalysis(name='buffalo_l')  # 'buffalo_l' is good balance of speed/accuracy
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 → CPU; use -1 for GPU if available

employees = {}  # Dictionary: { "Employee Name": normalized_embedding }

print(f"Scanning folder: {PHOTOS_DIR}")

for filename in os.listdir(PHOTOS_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Use filename without extension as employee name
        name = os.path.splitext(filename)[0].strip()
        filepath = os.path.join(PHOTOS_DIR, filename)
        
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error: Cannot read image {filename}")
            continue
        
        # Detect and extract face embedding
        faces = face_analyzer.get(img)
        
        if len(faces) == 0:
            print(f"Warning: No face detected in {filename}")
            continue
        
        if len(faces) > 1:
            print(f"Warning: Multiple faces in {filename} — using the first one")
        
        # Store normalized embedding (usually better for comparison)
        employees[name] = faces[0].normed_embedding
        print(f"Added: {name}")

# Create directory if it doesn't exist
os.makedirs(DATABASE_DIR, exist_ok=True)

# Save the database
with open(DATABASE_FILE, "wb") as f:
    pickle.dump(employees, f)

print(f"\nDone! Database saved to: {DATABASE_FILE}")
print(f"Total employees in database: {len(employees)}")
print("You can now run main.py")