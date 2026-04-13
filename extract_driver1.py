import cv2
import os

# --- CONFIGURATION ---
VIDEO_PATH = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\Harsha.mp4" 
OUTPUT_DIR = "Custom_Dataset/Driver_5"
FPS_TO_SAVE = 2 

# List of (Start_Min.Sec, End_Min.Sec, Sub_Folder_Name)
segments = [
    # Attentive
    ("0.00", "2.42", "attentive/c0_normal"),
    ("2.46", "3.26", "attentive/c0_normal"),
    ("3.29", "4.02", "attentive/c0_normal"),
    ("4.06", "4.16", "attentive/c0_normal"),
    ("4.27", "5.10", "attentive/c0_normal"),
    
    # Phone (Reels/Watching)
    ("5.16", "5.30", "distracted/c1_phone_reels"),
    ("5.36", "6.07", "distracted/c1_phone_reels"),
    ("6.12", "6.37", "distracted/c1_phone_reels"),
    ("6.43", "7.01", "distracted/c1_phone_reels"),
    ("7.06", "7.37", "distracted/c1_phone_reels"),
    ("7.43", "7.55", "distracted/c1_phone_reels"),
    ("8.02", "8.41", "distracted/c1_phone_reels"),
    
    # Drinking Water
    ("8.44", "9.10", "distracted/c6_drinking"),
    ("9.14", "9.57", "distracted/c6_drinking"),
    
    # Billboards
    ("9.58", "10.39", "distracted/c11_billboards"),
    ("10.44", "11.00", "distracted/c11_billboards"),
    ("11.07", "11.36", "distracted/c11_billboards"),
    
    # Passenger Talking
    ("11.42", "12.21", "distracted/c9_passenger"),
    ("12.25", "13.01", "distracted/c9_passenger"),
    ("13.06", "13.25", "distracted/c9_passenger"),
    ("13.33", "14.40", "distracted/c9_passenger"),
    
    # Drowsiness
    ("15.11", "15.22", "distracted/c10_drowsy"),
    ("15.28", "17.03", "distracted/c10_drowsy")
]

def min_sec_to_total_sec(ts):
    separator = ':' if ':' in ts else '.'
    parts = ts.split(separator)
    if len(parts) == 2:
        m, s = map(float, parts)
        return int(m * 60 + s)
    else:
        return int(float(parts[0]))

# --- EXECUTION ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: Could not open video at {VIDEO_PATH}")
else:
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    save_interval = max(1, int(actual_fps / FPS_TO_SAVE))

    frame_count = 0
    saved_count = 0

    print(f"🎥 Processing Video... Actual FPS: {actual_fps:.2f}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        curr_time = frame_count / actual_fps
        
        for start_ts, end_ts, subpath in segments:
            start_s = min_sec_to_total_sec(start_ts)
            end_s = min_sec_to_total_sec(end_ts)
            
            if start_s <= curr_time <= end_s:
                if frame_count % save_interval == 0:
                    # This creates: Custom_Dataset/Driver_4/distracted/c1_phone_reels/
                    folder = os.path.join(OUTPUT_DIR, subpath)
                    os.makedirs(folder, exist_ok=True)
                    
                    # Clean the label name for the filename
                    label_name = subpath.split('/')[-1]
                    filename = f"D4_{label_name}_{frame_count:06d}.jpg"
                    
                    cv2.imwrite(os.path.join(folder, filename), frame)
                    saved_count += 1
                break # Matched this frame to a segment, move to next frame
        
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"⏱️  At {int(curr_time // 60)}m {int(curr_time % 60)}s... Saved: {saved_count}")
        
    cap.release()
    print("-" * 50)
    print(f"✅ Finished Driver 4!")
    print(f"📂 Folders created inside: {OUTPUT_DIR}")
    print(f"📸 Total images saved: {saved_count}")