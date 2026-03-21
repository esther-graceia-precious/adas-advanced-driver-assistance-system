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
video_model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_model_86_best.h5",
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

def predict_combined(frame):
    img = preprocess(frame)

    # PRIMARY — video model on full frame
    video_prob = float(video_model.predict(img, verbose=0)[0][0])
    is_distracted_video = video_prob > 0.45
    # Additional override — if head is clearly down with high confidence
    # driver is likely looking at phone/object
    if not is_distracted_video:
        head_for_check = HEAD_CLASSES[np.argmax(
            head_model.predict(preprocess(get_face_crop(frame)[0] 
            if get_face_crop(frame)[0] is not None else frame), verbose=0)[0]
        )]

    # SECONDARY — multistream on face crop
    face_crop, face_coords = get_face_crop(frame)
    face_found = face_crop is not None

    if face_found:
        face_img   = preprocess(face_crop)
        head_pred  = head_model.predict(face_img,  verbose=0)[0]
        eye_pred   = eye_model.predict(face_img,   verbose=0)[0]
        mouth_pred = mouth_model.predict(face_img, verbose=0)[0]
    else:
        head_pred  = np.zeros(9)
        eye_pred   = np.zeros(6)
        mouth_pred = np.zeros(3)

    head_class  = HEAD_CLASSES[np.argmax(head_pred)]
    eye_class   = EYE_CLASSES[np.argmax(eye_pred)]
    mouth_class = MOUTH_CLASSES[np.argmax(mouth_pred)]
    head_conf   = float(np.max(head_pred))
    eye_conf    = float(np.max(eye_pred))
    mouth_conf  = float(np.max(mouth_pred))

    # REASONS — based on face crop analysis
    reasons = []
    if is_distracted_video and face_found:
        if head_class in ['Left', 'Right', 'Diagonal Up Left', 'Diagonal Up Right'] \
           and eye_class in ['Up', 'Left', 'Right']:
            reasons.append("Phone/Object use suspected")
        else:
            if head_class in DISTRACTED_HEAD:
                reasons.append(f"Head {head_class}")
            if eye_class == 'Closed':
                reasons.append("Eyes Closed (Fatigue)")
            elif eye_class in DISTRACTED_EYE:
                reasons.append(f"Gaze {eye_class}")
            if mouth_class == 'Wide Open':
                if head_class in ['Down', 'Diagonal Down Left', 'Diagonal Down Right']:
                    reasons.append("Drinking/Eating suspected")
                else:
                    reasons.append("Yawning (Fatigue)")

    if is_distracted_video and not reasons:
        reasons.append("Distraction detected")

    return {
        'is_distracted': is_distracted_video,
        'video_confidence': round(video_prob * 100, 1),
        'face_detected': face_found,
        'face_coords': face_coords,
        'head':  {'class': head_class if face_found else 'N/A', 'confidence': round(head_conf * 100, 1)},
        'eye':   {'class': eye_class  if face_found else 'N/A', 'confidence': round(eye_conf  * 100, 1)},
        'mouth': {'class': mouth_class if face_found else 'N/A','confidence': round(mouth_conf* 100, 1)},
        'reasons': reasons
    }