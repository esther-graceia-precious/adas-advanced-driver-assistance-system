# prepare_dataset.py
import cv2
import os

# ================================
# PATHS
# ================================
base = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Video Data"

# Normal Driving = Attentive
attentive_videos = os.path.join(base, "Normal Driving")

# All others = Distracted
distracted_folders = [
    os.path.join(base, "Electronic Media"),
    os.path.join(base, "Glancing at Billboards"),
    os.path.join(base, "Observing Scenery"),
    os.path.join(base, "Passenger Interaction"),
    os.path.join(base, "Rubbernecking")
]

# Output folders
out_attentive  = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset\attentive"
out_distracted = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset\distracted"

os.makedirs(out_attentive,  exist_ok=True)
os.makedirs(out_distracted, exist_ok=True)

# ================================
# EXTRACT FRAMES FUNCTION
# ================================
def extract_frames(video_path, output_folder, prefix, every_n=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Cannot open: {video_path}")
        return 0
    
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            filename = os.path.join(output_folder, f"{prefix}_{saved}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1
    cap.release()
    return saved

# ================================
# EXTRACT ATTENTIVE FRAMES
# ================================
print("\n📁 Extracting ATTENTIVE frames (Normal Driving)...")
total_attentive = 0
for video_name in os.listdir(attentive_videos):
    if not video_name.endswith('.mp4'):
        continue
    video_path = os.path.join(attentive_videos, video_name)
    prefix = f"att_{os.path.splitext(video_name)[0]}"
    saved = extract_frames(video_path, out_attentive, prefix, every_n=15)
    total_attentive += saved
    print(f"  ✅ {video_name}: {saved} frames")

print(f"\nTotal attentive frames: {total_attentive}")

# ================================
# EXTRACT DISTRACTED FRAMES
# ================================
print("\n📁 Extracting DISTRACTED frames...")
total_distracted = 0
for folder in distracted_folders:
    folder_name = os.path.basename(folder)
    print(f"\n  Folder: {folder_name}")
    for video_name in os.listdir(folder):
        if not video_name.endswith('.mp4'):
            continue
        video_path = os.path.join(folder, video_name)
        prefix = f"dis_{folder_name[:3]}_{os.path.splitext(video_name)[0]}"
        saved = extract_frames(video_path, out_distracted, prefix, every_n=15)
        total_distracted += saved
        print(f"    ✅ {video_name}: {saved} frames")

print(f"\nTotal distracted frames: {total_distracted}")
print(f"\n✅ DONE!")
print(f"Attentive : {total_attentive} frames")
print(f"Distracted: {total_distracted} frames")
