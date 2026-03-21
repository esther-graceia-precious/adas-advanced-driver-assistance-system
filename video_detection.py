# video_detection.py
import tensorflow as tf
import cv2
import numpy as np

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_model_best.h5",
    custom_objects={'loss': focal_loss()},
    compile=False
)
print("✅ Model loaded!")

# ================================
# PREPROCESS
# ================================
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

# ================================
# LOAD VIDEO
# ================================
cap = cv2.VideoCapture(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\VID20240305200614.mp4"
)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

print("▶ Video running. Press ESC to quit.")

distracted_counter = 0
ALERT_THRESHOLD = 15

# ================================
# MAIN LOOP
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    prob = float(model.predict(preprocess(frame), verbose=0)[0][0])

    if prob > 0.5:
        label = f"Distracted ({prob:.2f})"
        color = (0, 0, 255)
        distracted_counter += 1
    else:
        label = f"Safe ({prob:.2f})"
        color = (0, 255, 0)
        distracted_counter = 0

    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if distracted_counter >= ALERT_THRESHOLD:
        cv2.putText(frame, "!! ALERT: DISTRACTED !!",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("ADAS - Driver Distraction Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")