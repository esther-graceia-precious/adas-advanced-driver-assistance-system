# save_frames.py
import cv2
import os

os.makedirs("multistream/results/frames", exist_ok=True)

cap = cv2.VideoCapture(r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\results\output_detection.avi")

count = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Save every 30th frame
    if count % 30 == 0:
        cv2.imwrite(f"multistream/results/frames/frame_{saved}.jpg", frame)
        saved += 1
    count += 1

cap.release()
print(f"✅ Saved {saved} frames to multistream/results/frames/")