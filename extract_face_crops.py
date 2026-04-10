# extract_face_crops.py
import cv2
import os
import numpy as np
from tqdm import tqdm

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def get_face_crop(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        pad = 30
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2*pad)
        h = min(frame.shape[0] - y, h + 2*pad)
        return frame[y:y+h, x:x+w]
    return None

def extract_faces_from_folder(src, dst, label):
    os.makedirs(dst, exist_ok=True)
    images = [f for f in os.listdir(src)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]
    saved = 0
    skipped = 0
    print(f"\n📁 Processing {label}: {len(images)} images")
    for img_name in tqdm(images):
        frame = cv2.imread(os.path.join(src, img_name))
        if frame is None:
            continue
        face = get_face_crop(frame)
        if face is not None:
            cv2.imwrite(os.path.join(dst, img_name), face)
            saved += 1
        else:
            skipped += 1
    print(f"  ✅ Saved: {saved} | ❌ Skipped (no face): {skipped}")
    return saved

BASE = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT"

# Extract from SDDD video frames (your main training data)
extract_faces_from_folder(
    os.path.join(BASE, "new_dataset", "attentive"),
    os.path.join(BASE, "face_dataset", "attentive"),
    "SDDD Attentive"
)
extract_faces_from_folder(
    os.path.join(BASE, "new_dataset", "distracted"),
    os.path.join(BASE, "face_dataset", "distracted"),
    "SDDD Distracted"
)
print("\n✅ Face crop dataset ready!")