# evaluate_video_model.py
import cv2
import numpy as np
from app import model, get_multistream_info, preprocess
from collections import deque

def collect_probs(video_path):
    """Collect raw model probabilities from a video without any threshold decision."""
    cap   = cv2.VideoCapture(video_path)
    probs = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            prob = float(model.predict(preprocess(frame), verbose=0)[0][0])
            probs.append(prob)
        frame_count += 1

    cap.release()
    return probs


def benchmark_video(video_path, expected_label, threshold=0.5):
    cap             = cv2.VideoCapture(video_path)
    temporal_buffer = deque(maxlen=10)
    correct_frames  = 0
    total_samples   = 0
    counters        = {'eye': 0, 'fatigue': 0, 'dur': 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if total_samples % 5 == 0:
            prob = float(model.predict(preprocess(frame), verbose=0)[0][0])

            ms, counters['eye'], counters['fatigue'], counters['dur'] = get_multistream_info(
                frame, prob > threshold,
                counters['eye'], counters['fatigue'], counters['dur']
            )

            temporal_buffer.append(1 if ms['final_distracted'] else 0)
            smoothed_status = "Distracted" if sum(temporal_buffer) >= 6 else "Attentive"

            if smoothed_status == expected_label:
                correct_frames += 1

        total_samples += 1

    cap.release()
    sampled = total_samples // 5 or 1
    accuracy = (correct_frames / sampled) * 100
    return accuracy


def find_best_threshold(attentive_path, distracted_path):
    """
    Sweep thresholds from 0.30 to 0.90.
    Best threshold = highest balanced accuracy (avg of both videos).
    """
    print("\n📊 Collecting raw probabilities...")
    att_probs  = collect_probs(attentive_path)
    dis_probs  = collect_probs(distracted_path)

    print(f"\n--- Raw Probability Stats ---")
    print(f"Attentive  video → mean: {np.mean(att_probs):.3f}  "
          f"min: {np.min(att_probs):.3f}  max: {np.max(att_probs):.3f}")
    print(f"Distracted video → mean: {np.mean(dis_probs):.3f}  "
          f"min: {np.min(dis_probs):.3f}  max: {np.max(dis_probs):.3f}")

    thresholds = [round(t, 2) for t in np.arange(0.30, 0.91, 0.05)]
    best_threshold = 0.5
    best_balanced  = 0.0

    print(f"\n{'Threshold':<12} {'Attentive Acc':<18} {'Distracted Acc':<18} {'Balanced Acc'}")
    print("-" * 62)

    for t in thresholds:
        att_acc = benchmark_video(attentive_path,  "Attentive",  threshold=t)
        dis_acc = benchmark_video(distracted_path, "Distracted", threshold=t)
        balanced = (att_acc + dis_acc) / 2

        print(f"{t:<12.2f} {att_acc:<18.2f} {dis_acc:<18.2f} {balanced:.2f}")

        if balanced > best_balanced:
            best_balanced  = balanced
            best_threshold = t

    print(f"\n✅ Best Threshold : {best_threshold}")
    print(f"✅ Best Balanced Accuracy: {best_balanced:.2f}%")
    return best_threshold


# ================================
# MAIN
# ================================
ATTENTIVE_VIDEO  = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\Video Project_attentive _check.mp4"
DISTRACTED_VIDEO = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\Video Project_Check_dis.mp4"

print("=" * 62)
print("  ADAS Threshold Calibration — Webcam Domain")
print("=" * 62)

best_t = find_best_threshold(ATTENTIVE_VIDEO, DISTRACTED_VIDEO)

print("\n--- Final Verification at Best Threshold ---")
att_final = benchmark_video(ATTENTIVE_VIDEO,  "Attentive",  threshold=best_t)
dis_final = benchmark_video(DISTRACTED_VIDEO, "Distracted", threshold=best_t)
print(f"Attentive  Accuracy : {att_final:.2f}%")
print(f"Distracted Accuracy : {dis_final:.2f}%")
print(f"Balanced   Accuracy : {(att_final + dis_final) / 2:.2f}%")

print(f"""
→ Update your app.py live route:
    ai_distracted = prob > {best_t}   # webcam-calibrated threshold
""")