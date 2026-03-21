# multistream/test_normal.py
import cv2
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT")

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

video_model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_model_86_best.h5",
    custom_objects={'loss': focal_loss()}, compile=False
)
head_model  = tf.keras.models.load_model(r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\head_model_best.h5", compile=False)
eye_model   = tf.keras.models.load_model(r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\eye_model_best.h5",  compile=False)
mouth_model = tf.keras.models.load_model(r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\mouth_model_best.h5", compile=False)

HEAD_CLASSES  = ['Diagonal Down Left', 'Diagonal Down Right', 'Diagonal Up Left',
                 'Diagonal Up Right', 'Down', 'Frontal', 'Left', 'Right', 'Up']
EYE_CLASSES   = ['Closed', 'Down', 'Front', 'Left', 'Right', 'Up']
MOUTH_CLASSES = ['Closed', 'Slight Open', 'Wide Open']

cap = cv2.VideoCapture(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Video Data\Normal Driving\VID20240305200035.mp4"
)

print("\nTesting Normal Driving video - first 10 frames:\n")
count = 0
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    video_prob  = float(video_model.predict(img, verbose=0)[0][0])
    head_pred   = head_model.predict(img,  verbose=0)[0]
    eye_pred    = eye_model.predict(img,   verbose=0)[0]
    mouth_pred  = mouth_model.predict(img, verbose=0)[0]

    head_class  = HEAD_CLASSES[np.argmax(head_pred)]
    eye_class   = EYE_CLASSES[np.argmax(eye_pred)]
    mouth_class = MOUTH_CLASSES[np.argmax(mouth_pred)]

    print(f"Frame {i+1}:")
    print(f"  Video model:  {'DISTRACTED' if video_prob > 0.5 else 'ATTENTIVE'} ({video_prob*100:.1f}%)")
    print(f"  Head:         {head_class} ({np.max(head_pred)*100:.1f}%)")
    print(f"  Eye:          {eye_class}  ({np.max(eye_pred)*100:.1f}%)")
    print(f"  Mouth:        {mouth_class} ({np.max(mouth_pred)*100:.1f}%)")
    print()

cap.release()