# multistream/video_detection_multistream.py
import cv2
import sys
import os
sys.path.insert(0, r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT")
from multistream.fusion_combined import predict_combined

VIDEO_PATH = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Video Data\Electronic Media\electr1.mp4"
OUTPUT_PATH = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\results\output_detection.avi"

os.makedirs(r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\results", exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps    = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out    = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

print("Starting detection... Press ESC to quit")
frame_count = 0
last_result = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 5 == 0:
        last_result = predict_combined(frame)

    if last_result:
        r = last_result
        status = "DISTRACTED" if r['is_distracted'] else "ATTENTIVE"
        color  = (0, 0, 255) if r['is_distracted'] else (0, 255, 0)

        # Draw face box
        if r['face_detected'] and r['face_coords']:
            x, y, w, h = r['face_coords']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)

        # Info box
        cv2.rectangle(frame, (10, 10), (640, 210), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {status} ({r['video_confidence']}%)",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        face_txt = "Face: Detected" if r['face_detected'] else "Face: Not detected"
        cv2.putText(frame, face_txt, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 1)
        cv2.putText(frame, f"Head:  {r['head']['class']} ({r['head']['confidence']}%)",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Eye:   {r['eye']['class']} ({r['eye']['confidence']}%)",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Mouth: {r['mouth']['class']} ({r['mouth']['confidence']}%)",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if r['reasons']:
            cv2.putText(frame, f"Reason: {', '.join(r['reasons'])}",
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    out.write(frame)
    cv2.imshow("ADAS Multistream Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Saved to: {OUTPUT_PATH}")