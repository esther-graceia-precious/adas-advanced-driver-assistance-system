import cv2
import os

# --- CONFIGURATION ---
VIDEO_PATH = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\Kousik.mp4" # Path to your second video
OUTPUT_DIR = "Custom_Dataset/Driver_2"
FPS_TO_SAVE = 2 

# List of (Start_Min.Sec, End_Min.Sec, Class_Label)
segments = [
    # Attentive
    ("0.00", "2.31", "c0_attentive"),
    ("2.38", "3.39", "c0_attentive"),
    ("3.46", "4.00", "c0_attentive"),
    ("4.04", "4.26", "c0_attentive"),
    ("4.54", "5.01", "c0_attentive"),
    
    # Phone (Reels/Usage)
    ("5.13", "5.28", "c1_phone_reels"),
    ("5.32", "7.04", "c1_phone_reels"),
    
    # Drinking Water
    ("7.09", "7.20", "c6_drinking"),
    ("7.25", "7.32", "c6_drinking"),
    ("7.37", "7.45", "c6_drinking"),
    ("7.58", "8.05", "c6_drinking"),
    ("8.14", "8.18", "c6_drinking"),
    ("8.30", "8.59", "c6_drinking"),
    
    # Billboards
    ("9.04", "9.13", "c11_billboards"),
    ("9.23", "9.28", "c11_billboards"),
    ("9.41", "10.09", "c11_billboards"),
    ("10.19", "10.22", "c11_billboards"),
    ("10.27", "11.03", "c11_billboards"),
    
    # Phone (Talking)
    ("11.10", "11.52", "c2_phone_talking"),
    ("12.00", "12.44", "c2_phone_talking"),
    ("12.51", "13.03", "c2_phone_talking"), # Fixed the 1:03 to 13:03
    
    # Passenger Talking
    ("13.21", "13.36", "c9_passenger"),
    ("13.54", "14.07", "c9_passenger"),
    ("14.19", "14.27", "c9_passenger"),
    ("14.38", "14.58", "c9_passenger"),
    ("15.04", "15.14", "c9_passenger"),
    
    # Drowsiness
    ("15.52", "15.54", "c10_drowsy"),
    ("16.17", "16.32", "c10_drowsy")
]

def min_sec_to_total_sec(ts):
    # This handles both 2:31 and 2.31 automatically
    separator = ':' if ':' in ts else '.'
    parts = ts.split(separator)
    
    if len(parts) == 2:
        m, s = map(float, parts)
        return int(m * 60 + s)
    else:
        # If it's just a single number like "5", treat it as 5 seconds
        return int(float(parts[0]))

# --- EXECUTION ---
cap = cv2.VideoCapture(VIDEO_PATH)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
save_interval = int(actual_fps / FPS_TO_SAVE)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    curr_time = frame_count / actual_fps
    
    for start_ts, end_ts, label in segments:
        start_s = min_sec_to_total_sec(start_ts)
        end_s = min_sec_to_total_sec(end_ts)
        
        if start_s <= curr_time <= end_s:
            if frame_count % save_interval == 0:
                folder = os.path.join(OUTPUT_DIR, label)
                os.makedirs(folder, exist_ok=True)
                
                filename = f"D2_{label}_{frame_count}.jpg"
                cv2.imwrite(os.path.join(folder, filename), frame)
                saved_count += 1
    
    frame_count += 1
cap.release()
print(f"Finished Driver 2! Saved {saved_count} images.")