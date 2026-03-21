# fix_attentive.py
import cv2
import os

attentive_videos = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Video Data\Normal Driving"
out_attentive = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset\attentive"

def extract_frames(video_path, output_folder, prefix, every_n=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            filename = os.path.join(output_folder, f"fix_{prefix}_{saved}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1
    cap.release()
    return saved

print("Extracting more attentive frames...")
total = 0
for video_name in os.listdir(attentive_videos):
    if not video_name.endswith('.mp4'):
        continue
    video_path = os.path.join(attentive_videos, video_name)
    prefix = os.path.splitext(video_name)[0]
    saved = extract_frames(video_path, out_attentive, prefix, every_n=5)
    total += saved
    print(f"  ✅ {video_name}: {saved} frames")

print(f"\nNew total attentive frames: {total}")