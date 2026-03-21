# quick_test.py
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
    r'C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\video_model_nst_v2_final.h5',
    custom_objects={'loss': focal_loss()},
    compile=False
)
print('✅ Model loaded!')

def preprocess(frame):
    img = cv2.resize(frame, (224, 224)) / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

# ================================
# Test 1 - Normal Driving (should be ATTENTIVE)
# ================================
cap = cv2.VideoCapture(
    r'C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Video Data\Normal Driving\normal1.mp4'
)
print('\n--- Normal Driving (should be ATTENTIVE) ---')
correct = 0
for i in range(20):
    ret, frame = cap.read()
    if not ret:
        break
    prob = float(model.predict(preprocess(frame), verbose=0)[0][0])
    label = "Distracted" if prob > 0.5 else "Attentive"
    if label == "Attentive":
        correct += 1
    print(f'  Frame {i+1}: {label} ({prob:.3f})')
cap.release()
print(f'  Score: {correct}/20 correct')

# ================================
# Test 2 - Electronic Media (should be DISTRACTED)
# ================================
cap = cv2.VideoCapture(
    r'C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Video Data\Electronic Media\electr1.mp4'
)
print('\n--- Electronic Media (should be DISTRACTED) ---')
correct = 0
for i in range(20):
    ret, frame = cap.read()
    if not ret:
        break
    prob = float(model.predict(preprocess(frame), verbose=0)[0][0])
    label = "Distracted" if prob > 0.5 else "Attentive"
    if label == "Distracted":
        correct += 1
    print(f'  Frame {i+1}: {label} ({prob:.3f})')
cap.release()
print(f'  Score: {correct}/20 correct')

# ================================
# Test 3 - Passenger Interaction (should be DISTRACTED)
# ================================
cap = cv2.VideoCapture(
    r'C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Video Data\Passenger Interaction\passen1.mp4'
)
print('\n--- Passenger Interaction (should be DISTRACTED) ---')
correct = 0
for i in range(20):
    ret, frame = cap.read()
    if not ret:
        break
    prob = float(model.predict(preprocess(frame), verbose=0)[0][0])
    label = "Distracted" if prob > 0.5 else "Attentive"
    if label == "Distracted":
        correct += 1
    print(f'  Frame {i+1}: {label} ({prob:.3f})')
cap.release()
print(f'  Score: {correct}/20 correct')

print('\n✅ Test complete!')