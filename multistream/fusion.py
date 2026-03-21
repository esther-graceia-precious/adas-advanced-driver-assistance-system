# multistream/fusion_combined.py
import tensorflow as tf
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================================
# LOAD ALL MODELS
# ================================
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

print("Loading models...")
video_model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_model_86_best.h5",
    custom_objects={'loss': focal_loss()}, compile=False
)
head_model  = tf.keras.models.load_model('multistream/head_model_best.h5',  compile=False)
eye_model   = tf.keras.models.load_model('multistream/eye_model_best.h5',   compile=False)
mouth_model = tf.keras.models.load_model('multistream/mouth_model_best.h5', compile=False)
print("✅ All 4 models loaded!")

HEAD_CLASSES  = ['Diagonal Down Left', 'Diagonal Down Right', 'Diagonal Up Left',
                 'Diagonal Up Right', 'Down', 'Frontal', 'Left', 'Right', 'Up']
EYE_CLASSES   = ['Closed', 'Down', 'Front', 'Left', 'Right', 'Up']
MOUTH_CLASSES = ['Closed', 'Slight Open', 'Wide Open']

DISTRACTED_HEAD  = ['Left', 'Right', 'Down', 'Diagonal Down Left',
                    'Diagonal Down Right', 'Diagonal Up Left', 'Diagonal Up Right']
DISTRACTED_EYE   = ['Closed', 'Down', 'Left', 'Right', 'Up']
DISTRACTED_MOUTH = ['Wide Open']

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def predict_combined(frame):
    img = preprocess(frame)

    # Step 1 — Video model (binary)
    video_prob = float(video_model.predict(img, verbose=0)[0][0])
    is_distracted_video = video_prob > 0.5

    # Step 2 — Multistream models
    head_pred   = head_model.predict(img,  verbose=0)[0]
    eye_pred    = eye_model.predict(img,   verbose=0)[0]
    mouth_pred  = mouth_model.predict(img, verbose=0)[0]

    head_class  = HEAD_CLASSES[np.argmax(head_pred)]
    eye_class   = EYE_CLASSES[np.argmax(eye_pred)]
    mouth_class = MOUTH_CLASSES[np.argmax(mouth_pred)]

    # Step 3 — Find reasons
    reasons = []
    if head_class in DISTRACTED_HEAD:
        reasons.append(f"Head {head_class}")
    if eye_class in DISTRACTED_EYE:
        if eye_class == 'Closed':
            reasons.append("Eyes Closed (Fatigue)")
        else:
            reasons.append(f"Gaze {eye_class}")
    if mouth_class in DISTRACTED_MOUTH:
        reasons.append("Yawning (Fatigue)")

    # Step 4 — Final fusion decision
    if is_distracted_video and not reasons:
        reasons = ["Phone/Object use suspected"]

    final_distracted = is_distracted_video

    return {
        'is_distracted': final_distracted,
        'video_confidence': round(video_prob * 100, 1),
        'head':  {'class': head_class,  'confidence': round(float(np.max(head_pred))  * 100, 1)},
        'eye':   {'class': eye_class,   'confidence': round(float(np.max(eye_pred))   * 100, 1)},
        'mouth': {'class': mouth_class, 'confidence': round(float(np.max(mouth_pred)) * 100, 1)},
        'reasons': reasons
    }