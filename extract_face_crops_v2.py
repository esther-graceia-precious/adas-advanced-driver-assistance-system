# extract_face_crops_v2.py
# Uses OpenCV DNN face detector (res10 SSD) instead of Haar cascade
# Much better at detecting non-frontal faces (looking down, sideways etc.)
# No new package installs needed — just 2 model files downloaded once

import cv2
import os
import numpy as np
from tqdm import tqdm

BASE = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT"

# -------------------------------------------------------------------
# Load OpenCV DNN face detector (res10 SSD)
# Download these two files into your ADAS_PROJECT folder:
# 1. deploy.prototxt
# 2. res10_300x300_ssd_iter_140000.caffemodel
# -------------------------------------------------------------------
PROTO = os.path.join(BASE, "deploy.prototxt")
MODEL = os.path.join(BASE, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(PROTO) or not os.path.exists(MODEL):
    raise FileNotFoundError(
        "Missing DNN model files. Download:\n"
        "  deploy.prototxt\n"
        "  res10_300x300_ssd_iter_140000.caffemodel\n"
        f"Place both in: {BASE}"
    )

net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)


def get_face_crop_dnn(frame, confidence_threshold=0.4, pad=40):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0
    best_box = None

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < confidence_threshold:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        if conf > best_conf:
            best_conf = conf
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def extract_faces_from_folder(src, dst, label):
    os.makedirs(dst, exist_ok=True)
    images = [f for f in os.listdir(src)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    saved = 0
    skipped = 0
    print(f"\n📁 Processing {label}: {len(images)} images")
    for img_name in tqdm(images):
        frame = cv2.imread(os.path.join(src, img_name))
        if frame is None:
            continue
        face = get_face_crop_dnn(frame)
        if face is not None:
            cv2.imwrite(os.path.join(dst, img_name), face)
            saved += 1
        else:
            skipped += 1
    print(f"  ✅ Saved: {saved} | ❌ Skipped (no face): {skipped}")
    return saved


if __name__ == "__main__":
    extract_faces_from_folder(
        os.path.join(BASE, "new_dataset", "attentive"),
        os.path.join(BASE, "face_dataset_v2", "attentive"),
        "SDDD Attentive"
    )
    extract_faces_from_folder(
        os.path.join(BASE, "new_dataset", "distracted"),
        os.path.join(BASE, "face_dataset_v2", "distracted"),
        "SDDD Distracted"
    )
    print("\n✅ face_dataset_v2 ready! Retrain train_face_model.py pointing to face_dataset_v2")