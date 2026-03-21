# combine_datasets.py
import os
import shutil

base = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT"

# ================================
# DESTINATIONS
# ================================
att_dst = os.path.join(base, "new_dataset", "attentive")
dis_dst = os.path.join(base, "new_dataset", "distracted")
os.makedirs(att_dst, exist_ok=True)
os.makedirs(dis_dst, exist_ok=True)

def copy_images(src_folder, dst_folder, prefix):
    if not os.path.exists(src_folder):
        print(f"  ⚠️ Not found: {src_folder}")
        return 0
    count = 0
    for img in os.listdir(src_folder):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            dst_file = os.path.join(dst_folder, f"{prefix}_{count}.jpg")
            shutil.copy2(os.path.join(src_folder, img), dst_file)
            count += 1
    return count

total_att = 0
total_dis = 0

# ================================
# 1. ORIGINAL DATASET
# ================================
print("\n📁 Adding original dataset...")
n = copy_images(
    os.path.join(base, "dataset", "attentive"),
    att_dst, "orig_att"
)
total_att += n
print(f"  ✅ Original attentive: {n} images")

n = copy_images(
    os.path.join(base, "dataset", "distracted"),
    dis_dst, "orig_dis"
)
total_dis += n
print(f"  ✅ Original distracted: {n} images")

# ================================
# 2. SDDD HEAD POSE
# ================================
print("\n📁 Adding SDDD Head Pose...")
head_pose_base = os.path.join(base, "dataset", "Image Data", "Head Pose")

# Attentive - Frontal only
n = copy_images(
    os.path.join(head_pose_base, "Frontal"),
    att_dst, "head_frontal"
)
total_att += n
print(f"  ✅ Head Frontal (attentive): {n} images")

# Distracted - all other poses
distracted_poses = [
    "Left", "Right", "Up", "Down",
    "Diagonal Down Left", "Diagonal Down Right",
    "Diagonal Up Left", "Diagonal Up Right"
]
for pose in distracted_poses:
    n = copy_images(
        os.path.join(head_pose_base, pose),
        dis_dst, f"head_{pose.replace(' ', '_')}"
    )
    total_dis += n
    print(f"  ✅ Head {pose} (distracted): {n} images")

# ================================
# 3. SDDD EYE GAZE
# ================================
print("\n📁 Adding SDDD Eye Gaze...")
eye_base = os.path.join(base, "dataset", "Image Data", "Eye Gaze")

# Attentive - Front only
n = copy_images(
    os.path.join(eye_base, "Front"),
    att_dst, "eye_front"
)
total_att += n
print(f"  ✅ Eye Front (attentive): {n} images")

# Distracted - all others
distracted_eyes = ["Left", "Right", "Up", "Down", "Closed"]
for eye in distracted_eyes:
    n = copy_images(
        os.path.join(eye_base, eye),
        dis_dst, f"eye_{eye}"
    )
    total_dis += n
    print(f"  ✅ Eye {eye} (distracted): {n} images")

# ================================
# 4. SDDD MOUTH STATES
# ================================
print("\n📁 Adding SDDD Mouth States...")
mouth_base = os.path.join(base, "dataset", "Image Data", "Mouth States")

# Attentive - Closed only
n = copy_images(
    os.path.join(mouth_base, "Closed"),
    att_dst, "mouth_closed"
)
total_att += n
print(f"  ✅ Mouth Closed (attentive): {n} images")

# Distracted - Slight Open, Wide Open
for state in ["Slight Open", "Wide Open"]:
    n = copy_images(
        os.path.join(mouth_base, state),
        dis_dst, f"mouth_{state.replace(' ', '_')}"
    )
    total_dis += n
    print(f"  ✅ Mouth {state} (distracted): {n} images")

# ================================
# SUMMARY
# ================================
print(f"\n{'='*40}")
print(f"✅ DONE!")
print(f"Total Attentive : {total_att} images")
print(f"Total Distracted: {total_dis} images")
print(f"{'='*40}")
print(f"\nNow run: python train_video_model.py")