# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os
import sys
import tempfile
import base64

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000"
])

# ================================
# LOAD MAIN VIDEO MODEL
# ================================
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "multistream", "multistream", "video_model_nst_v2_best.h5")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'loss': focal_loss()},
    compile=False
)
print("✅ Main model loaded!")

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

last_conv_layer = get_last_conv_layer(model)
print(f"✅ Last conv layer: {last_conv_layer}")

# ================================
# LOAD MULTISTREAM MODELS
# ================================
HEAD_CLASSES  = ['Diagonal Down Left', 'Diagonal Down Right', 'Diagonal Up Left',
                 'Diagonal Up Right', 'Down', 'Frontal', 'Left', 'Right', 'Up']
EYE_CLASSES   = ['Closed', 'Down', 'Front', 'Left', 'Right', 'Up']
MOUTH_CLASSES = ['Closed', 'Slight Open', 'Wide Open']

DISTRACTED_HEAD  = ['Left', 'Right', 'Down', 'Diagonal Down Left',
                    'Diagonal Down Right', 'Diagonal Up Left', 'Diagonal Up Right']
DISTRACTED_EYE   = ['Closed', 'Down', 'Left', 'Right', 'Up']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

try:
    head_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "multistream", "multistream", "head_model_best.h5"),
        compile=False
    )
    eye_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "multistream", "multistream", "eye_model_best.h5"),
        compile=False
    )
    mouth_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "multistream", "multistream", "mouth_model_best.h5"),
        compile=False
    )
    multistream_loaded = True
    print("✅ Multistream models loaded!")
except Exception as e:
    multistream_loaded = False
    print(f"⚠️ Multistream models not loaded: {e}")

# ================================
# HELPER FUNCTIONS
# ================================
def generate_gradcam(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def apply_gradcam_overlay(frame, heatmap):
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
    return overlay

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def get_risk_level(distracted_pct):
    if distracted_pct < 20:
        return "LOW"
    elif distracted_pct < 50:
        return "MEDIUM"
    else:
        return "HIGH"

def get_face_crop(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Equalize histogram for better detection in different lighting
    gray = cv2.equalizeHist(gray)
    
    for scale in [1.05, 1.1, 1.2, 1.3]:
        for neighbors in [3, 2, 1]:
            faces = face_cascade.detectMultiScale(
                gray, scale, neighbors,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                # Get largest face
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                pad = 25
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(frame.shape[1] - x, w + 2*pad)
                h = min(frame.shape[0] - y, h + 2*pad)
                return frame[y:y+h, x:x+w], (x, y, w, h)
    return None, None

def get_multistream_info(frame, is_distracted):
    """Get head/eye/mouth predictions and reasons"""
    if not multistream_loaded:
        return {
            'face_detected': False,
            'head': 'N/A', 'eye': 'N/A', 'mouth': 'N/A',
            'reasons': []
        }

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
        head_conf   = round(float(np.max(head_pred)) * 100, 1)
        eye_conf    = round(float(np.max(eye_pred))  * 100, 1)
        mouth_conf  = round(float(np.max(mouth_pred))* 100, 1)
    else:
        head_class = eye_class = mouth_class = 'N/A'
        head_conf  = eye_conf  = mouth_conf  = 0.0

    # Reasons
    reasons = []
    if is_distracted and face_found:
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

    if is_distracted and not reasons:
        reasons.append("Distraction detected")

    return {
        'face_detected': face_found,
        'head':  f"{head_class} ({head_conf}%)" if face_found else 'N/A',
        'eye':   f"{eye_class} ({eye_conf}%)"  if face_found else 'N/A',
        'mouth': f"{mouth_class} ({mouth_conf}%)" if face_found else 'N/A',
        'reasons': reasons
    }

# ================================
# IMAGE ANALYSIS
# ================================
@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    img_array  = np.frombuffer(image_file.read(), np.uint8)
    frame      = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Cannot read image'}), 400

    # Try face crop first to reduce background bias
    face_crop, _ = get_face_crop(frame)
    img_input = preprocess(face_crop if face_crop is not None else frame)
    prob      = float(model.predict(img_input, verbose=0)[0][0])
    label     = "Distracted" if prob > 0.5 else "Attentive"
    risk      = get_risk_level(prob * 100)

    # Multistream info
    ms_info = get_multistream_info(frame, prob > 0.5)

    # Grad-CAM
    gradcam_data = None
    try:
        heatmap  = generate_gradcam(img_input, model, last_conv_layer)
        overlay  = apply_gradcam_overlay(frame, heatmap)
        gradcam_data = {
            'original':   frame_to_base64(frame),
            'gradcam':    frame_to_base64(overlay),
            'confidence': round(prob * 100, 1)
        }
    except Exception as e:
        print(f"Grad-CAM error: {e}")

    return jsonify({
        'label':       label,
        'confidence':  round(prob * 100, 1),
        'risk_level':  risk,
        'multistream': ms_info,
        'gradcam':     gradcam_data
    })

# ================================
# VIDEO ANALYSIS
# ================================
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video_file = request.files['video']
    tmp        = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(tmp.name)
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video'}), 400

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0

    confidences      = []
    labels           = []
    timestamps       = []
    frame_reasons    = []
    frame_head       = []
    frame_eye        = []
    frame_mouth      = []
    processed_frames = []
    frame_count      = 0
    sample_every     = max(1, int(fps / 2))
    max_prob         = 0
    most_distracted_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_every == 0:
            img_input    = preprocess(frame)
            prob         = float(model.predict(img_input, verbose=0)[0][0])
            is_distracted = prob > 0.5
            label        = "Distracted" if is_distracted else "Attentive"

            confidences.append(round(prob, 4))
            labels.append(label)
            timestamps.append(round(frame_count / fps, 2) if fps > 0 else frame_count)

            # Multistream info
            ms = get_multistream_info(frame, is_distracted)
            frame_reasons.append(ms['reasons'])
            frame_head.append(ms['head'])
            frame_eye.append(ms['eye'])
            frame_mouth.append(ms['mouth'])

            # Draw overlay on frame
            color = (0, 0, 255) if is_distracted else (0, 255, 0)
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (10, 10), (550, 150), (0, 0, 0), -1)
            cv2.putText(display_frame, f"{label} ({prob*100:.1f}%)",
                       (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if ms['face_detected']:
                cv2.putText(display_frame, f"Head: {ms['head']}",
                           (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(display_frame, f"Eye:  {ms['eye']}",
                           (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(display_frame, f"Mouth:{ms['mouth']}",
                           (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            if ms['reasons']:
                cv2.putText(display_frame,
                           f"Reason: {', '.join(ms['reasons'])}",
                           (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

            if len(processed_frames) < 50:
                processed_frames.append(frame_to_base64(display_frame))

            if prob > max_prob:
                max_prob = prob
                most_distracted_frame = frame.copy()

        frame_count += 1

    cap.release()
    os.unlink(tmp.name)

    # Summary
    total            = len(labels)
    distracted_count = labels.count("Distracted")
    attentive_count  = labels.count("Attentive")
    distracted_pct   = round((distracted_count / total) * 100, 1) if total > 0 else 0
    attentive_pct    = round((attentive_count  / total) * 100, 1) if total > 0 else 0
    risk_level       = get_risk_level(distracted_pct)

    # Alerts
    alerts       = []
    in_alert     = False
    alert_start  = 0
    ALERT_THRESHOLD = 5
    consecutive  = 0

    for i, label in enumerate(labels):
        if label == "Distracted":
            consecutive += 1
            if consecutive >= ALERT_THRESHOLD and not in_alert:
                in_alert    = True
                alert_start = timestamps[i - ALERT_THRESHOLD + 1]
        else:
            if in_alert:
                alerts.append({'start': alert_start, 'end': timestamps[i - 1]})
            in_alert    = False
            consecutive = 0

    if in_alert:
        alerts.append({'start': alert_start, 'end': timestamps[-1]})

    # Grad-CAM
    gradcam_data = None
    if most_distracted_frame is not None:
        try:
            img_input      = preprocess(most_distracted_frame)
            heatmap        = generate_gradcam(img_input, model, last_conv_layer)
            gradcam_overlay = apply_gradcam_overlay(most_distracted_frame, heatmap)
            gradcam_data   = {
                'original':   frame_to_base64(most_distracted_frame),
                'gradcam':    frame_to_base64(gradcam_overlay),
                'confidence': round(max_prob * 100, 1)
            }
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    return jsonify({
        'summary': {
            'total_frames_analyzed': total,
            'duration_seconds':      round(duration, 2),
            'attentive_pct':         attentive_pct,
            'distracted_pct':        distracted_pct,
            'alert_count':           len(alerts),
            'overall_status':        'DISTRACTED' if distracted_pct > 50 else 'ATTENTIVE',
            'risk_level':            risk_level
        },
        'frame_data': {
            'timestamps':  timestamps,
            'confidences': confidences,
            'labels':      labels
        },
        'multistream': {
            'reasons': frame_reasons,
            'head':    frame_head,
            'eye':     frame_eye,
            'mouth':   frame_mouth
        },
        'processed_frames': processed_frames,
        'alerts':    alerts,
        'gradcam':   gradcam_data
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)