# multistream/neural_style_augment_v2.py
import numpy as np
import cv2
import os
import random
from tqdm import tqdm

def fast_style_augment(content_img, style_img):
    content = content_img.astype(np.float32) / 255.0
    style   = style_img.astype(np.float32) / 255.0
    result  = np.zeros_like(content)
    for c in range(3):
        c_mean = np.mean(content[:,:,c])
        c_std  = np.std(content[:,:,c]) + 1e-5
        s_mean = np.mean(style[:,:,c])
        s_std  = np.std(style[:,:,c]) + 1e-5
        normalized = (content[:,:,c] - c_mean) / c_std
        result[:,:,c] = normalized * s_std + s_mean
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

# ================================
# USE VIDEO FRAMES AS CONTENT
# ================================
VIDEO_ATTENTIVE  = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset\attentive"
VIDEO_DISTRACTED = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset\distracted"
OUTPUT_ATT       = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_augmented\attentive"
OUTPUT_DIS       = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_augmented\distracted"

os.makedirs(OUTPUT_ATT, exist_ok=True)
os.makedirs(OUTPUT_DIS, exist_ok=True)

# Load style images from original dataset (studio images as style)
STYLE_FOLDER = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\attentive"
style_images = []
for img_name in random.sample(os.listdir(STYLE_FOLDER), min(200, len(os.listdir(STYLE_FOLDER)))):
    if img_name.lower().endswith(('.jpg','.jpeg','.png')):
        img = cv2.imread(os.path.join(STYLE_FOLDER, img_name))
        if img is not None:
            style_images.append(cv2.resize(img, (224, 224)))

print(f"✅ Loaded {len(style_images)} style images")

def augment_folder(src, dst, prefix, ratio=0.5):
    images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"\n📁 Processing {prefix}: {len(images)} images")
    total = 0
    for img_name in tqdm(images):
        img = cv2.imread(os.path.join(src, img_name))
        if img is None: continue
        img = cv2.resize(img, (224, 224))

        # Copy original
        cv2.imwrite(os.path.join(dst, img_name), img)
        total += 1

        # Add augmented version
        if random.random() < ratio:
            style = random.choice(style_images)
            aug   = fast_style_augment(img, style)
            cv2.imwrite(os.path.join(dst, f"aug_{img_name}"), aug)
            total += 1
    return total

att_count = augment_folder(VIDEO_ATTENTIVE, OUTPUT_ATT, "attentive", ratio=0.5)
dis_count = augment_folder(VIDEO_DISTRACTED, OUTPUT_DIS, "distracted", ratio=0.3)

print(f"\n✅ Done!")
print(f"Attentive:  {att_count}")
print(f"Distracted: {dis_count}")