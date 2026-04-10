# multistream/fusion_combined.py
import tensorflow as tf
import cv2
import numpy as np

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

print("Loading models...")
# Change this line
video_model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\video_model_nst_final.h5",
    custom_objects={'loss': focal_loss()}, compile=False
)
head_model  = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\head_model_best.h5",
    compile=False
)
eye_model   = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\eye_model_best.h5",
    compile=False
)
mouth_model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\mouth_model_best.h5",
    compile=False
)
print("✅ All 4 models loaded!")

HEAD_CLASSES  = ['Diagonal Down Left', 'Diagonal Down Right', 'Diagonal Up Left',
                 'Diagonal Up Right', 'Down', 'Frontal', 'Left', 'Right', 'Up']
EYE_CLASSES   = ['Closed', 'Down', 'Front', 'Left', 'Right', 'Up']
MOUTH_CLASSES = ['Closed', 'Slight Open', 'Wide Open']
DISTRACTED_HEAD  = ['Left', 'Right', 'Down', 'Diagonal Down Left',
                    'Diagonal Down Right', 'Diagonal Up Left', 'Diagonal Up Right']
DISTRACTED_EYE   = ['Closed', 'Down', 'Left', 'Right', 'Up']

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def get_face_crop(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2*pad)
        h = min(frame.shape[0] - y, h + 2*pad)
        return frame[y:y+h, x:x+w], (x, y, w, h)
    return None, None

# --- ADD THESE GLOBALS FOR FATIGUE TRACKING ---
EYE_CLOSED_COUNTER = 0 
FATIGUE_THRESHOLD = 8 # Number of frames to trigger fatigue

def predict_combined(frame):
    global EYE_CLOSED_COUNTER
    img = preprocess(frame)

    # 1. PRIMARY — video model on full frame
    video_prob = float(video_model.predict(img, verbose=0)[0][0])
    
    # 2. SECONDARY — multistream on face crop
    face_crop, face_coords = get_face_crop(frame)
    face_found = face_crop is not None

    if face_found:
        face_img   = preprocess(face_crop)
        head_pred  = head_model.predict(face_img,  verbose=0)[0]
        eye_pred   = eye_model.predict(face_img,   verbose=0)[0]
        mouth_pred = mouth_model.predict(face_img, verbose=0)[0]
        
        head_class  = HEAD_CLASSES[np.argmax(head_pred)]
        eye_class   = EYE_CLASSES[np.argmax(eye_pred)]
        mouth_class = MOUTH_CLASSES[np.argmax(mouth_pred)]
    else:
        head_class = eye_class = mouth_class = 'N/A'

    # --- 3. GEOMETRIC HEURISTIC OVERRIDES (The Fix) ---
    heuristic_distracted = False
    reasons = []

    if face_found:
        # A. Phone/Lap Gaze: If head is down AND eyes are down/closed
        if head_class in ['Down', 'Diagonal Down Left', 'Diagonal Down Right'] and eye_class in ['Down', 'Closed']:
            heuristic_distracted = True
            reasons.append("Phone Use / Looking at Lap")

        # B. Fatigue: Check for consecutive eye closure
        if eye_class == 'Closed':
            EYE_CLOSED_COUNTER += 1
            if EYE_CLOSED_COUNTER >= FATIGUE_THRESHOLD:
                heuristic_distracted = True
                reasons.append("FATIGUE: Eyes Closed")
        else:
            EYE_CLOSED_COUNTER = 0

        # C. Visual Distraction: Head turned but eyes not front
        if head_class in ['Left', 'Right'] and eye_class != 'Front':
            heuristic_distracted = True
            reasons.append(f"Looking {head_class}")

    # Final Decision Fusion
    # We trust the heuristic (Logic) over the video model (Intuition)
    is_distracted = (video_prob > 0.55) or heuristic_distracted

    return {
        'is_distracted': is_distracted,
        'video_confidence': round(video_prob * 100, 1),
        'face_detected': face_found,
        'face_coords': face_coords,
        'head': {'class': head_class, 'confidence': 0}, # Simplified
        'eye':  {'class': eye_class, 'confidence': 0},
        'mouth': {'class': mouth_class, 'confidence': 0},
        'reasons': reasons
    }