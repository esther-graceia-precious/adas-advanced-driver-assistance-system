# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os
import tempfile
import base64
from collections import deque
import uuid
import requests as http_requests
import requests


app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000"
])

# ================================
# FACE DETECTION — DNN + Haar fallback
# ================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

_DNN_PROTO  = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\deploy.prototxt"
_DNN_MODEL  = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\res10_300x300_ssd_iter_140000.caffemodel"
_dnn_face_net = None

def _get_dnn_net():
    global _dnn_face_net
    if _dnn_face_net is None:
        if os.path.exists(_DNN_PROTO) and os.path.exists(_DNN_MODEL):
            try:
                _dnn_face_net = cv2.dnn.readNetFromCaffe(_DNN_PROTO, _DNN_MODEL)
                print("✅ DNN face detector loaded successfully")
            except Exception as e:
                print(f"❌ Error loading DNN: {e}")
        else:
            print(f"⚠️ DNN files NOT found at: {_DNN_PROTO}")
    return _dnn_face_net

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

DISTRACTED_HEAD = ['Left', 'Right', 'Down', 'Diagonal Down Left',
                   'Diagonal Down Right', 'Diagonal Up Left', 'Diagonal Up Right']

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
# LIVE SESSION STORE
# ================================
FATIGUE_THRESHOLD = 12  # ~2 seconds at ~6fps

live_sessions = {}

def new_session_state():
    return {
        'eye_closed_counter':   0,
        'fatigue_events_count': 0,
        'fatigue_max_duration': 0,
        'temporal_buffer':      deque(maxlen=15),  # 15-frame window for live
        'distracted_count':     0,
        'total_count':          0,
        'phone_suspect_counter': 0,   # ← ADD THIS
    }

# ================================
# HELPERS
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
    return cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

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
    
YOLO_API = "http://localhost:5001"
_yolo_available = None

def detect_objects_yolo(frame_b64):
    try:
        res = http_requests.post(
            f"{YOLO_API}/detect",
            json={'image': frame_b64},
            timeout=0.8
        )
        if res.status_code == 200:
            return res.json()
        return {'phone': False, 'drinking': False}
    except Exception:
        return {'phone': False, 'drinking': False}
    
def get_ear_mar(frame_b64):
    try:
        res = requests.post(
            "http://127.0.0.1:5002/analyze",
            json={"image": frame_b64},
            timeout=0.3
        )
        return res.json()
    except:
        return {'ear': None, 'mar': None}

def get_face_crop(frame):
    h, w = frame.shape[:2]
    net  = _get_dnn_net()

    if net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()
        best_conf  = 0
        best_box   = None
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.5:
                continue
            if conf > best_conf:
                best_conf = conf
                best_box  = (
                    int(detections[0, 0, i, 3] * w),
                    int(detections[0, 0, i, 4] * h),
                    int(detections[0, 0, i, 5] * w),
                    int(detections[0, 0, i, 6] * h),
                )
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            pad = 20
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                return crop, (x1, y1, x2 - x1, y2 - y1)

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        return frame[y:y + fh, x:x + fw], (x, y, fw, fh)
    return None, None

def calculate_safety_grade(distracted_pct, fatigue_events):
    score = 100 - distracted_pct - (fatigue_events * 15)
    score = max(0, round(score, 1))
    if score >= 90:
        return score, "A", "Excellent — Focused"
    if score >= 75:
        return score, "B", "Good — Minor Gaps"
    if score >= 60:
        return score, "C", "Caution — Frequent Distraction"
    return score, "F", "DANGEROUS — Immediate Risk"

def get_multistream_info(frame, ai_distracted,
                         eye_closed_counter, fatigue_events_count, fatigue_max_duration,
                         phone_suspect_counter=0,
                         live_mode=False,
                         frame_b64=None):
    if not multistream_loaded:
        return (
            {'face_detected': False, 'head': 'N/A', 'eye': 'N/A', 'mouth': 'N/A',
             'reasons': [], 'final_distracted': ai_distracted},
            eye_closed_counter, fatigue_events_count, fatigue_max_duration
        )

    face_crop, _ = get_face_crop(frame)
    face_found   = face_crop is not None

    if face_found:
        face_img    = preprocess(face_crop)
        h_preds     = head_model.predict(face_img, verbose=0)[0]
        e_preds     = eye_model.predict(face_img,  verbose=0)[0]
        m_preds     = mouth_model.predict(face_img, verbose=0)[0]

        h_conf      = np.max(h_preds)
        head_class  = HEAD_CLASSES[np.argmax(h_preds)] if h_conf >= 0.75 else 'Frontal'
        eye_class   = EYE_CLASSES[np.argmax(e_preds)]
        eye_conf    = np.max(e_preds)
        mouth_class = MOUTH_CLASSES[np.argmax(m_preds)]

        reasons              = []
        heuristic_distracted = False

        # Live uses 0.65 (more sensitive); video uses 0.70
        eye_conf_threshold = 0.65 if live_mode else 0.70

        # FATIGUE ACCUMULATOR
        if eye_class == 'Closed' and eye_conf > eye_conf_threshold:
            eye_closed_counter += 1
            if eye_closed_counter == FATIGUE_THRESHOLD:
                fatigue_events_count += 1
                reasons.append("Fatigue Detected")
            if eye_closed_counter >= FATIGUE_THRESHOLD:
                heuristic_distracted = True
                fatigue_max_duration = max(fatigue_max_duration, eye_closed_counter)
            # In live mode, even short eye closure (3+ frames) is flagged
            if live_mode and eye_closed_counter >= 3:
                heuristic_distracted = True
                if "Eyes Closed - Fatigue/Distraction" not in reasons:
                    reasons.append("Eyes Closed - Fatigue/Distraction")
        else:
            eye_closed_counter = 0

        # --- DISTRACTION BEHAVIOR DETECTION ---
        # Uses persistence to separate quick glances from sustained distraction

        # PHONE USAGE: head turned + sustained (not a quick mirror check)
        # Pass phone_suspect_counter in and out like eye_closed_counter
        if head_class in ['Left', 'Right'] and eye_class in ['Left', 'Right', 'Front']:
            phone_suspect_counter += 1
            if phone_suspect_counter >= 5:   # ~5 frames sustained = phone not mirror
                if "Phone Usage Suspected" not in reasons:
                    reasons.append("Phone Usage Suspected")
        else:
            phone_suspect_counter = 0

        # DRINKING: head down + mouth wide open
        if head_class == 'Down' and mouth_class == 'Wide Open':
            if "Drinking Suspected" not in reasons:
                reasons.append("Drinking Suspected")

        # EATING: mouth wide open while roughly frontal (not looking away)
        if mouth_class == 'Wide Open' and head_class in ['Frontal', 'Down']:
            if "Eating Suspected" not in reasons:
                reasons.append("Eating Suspected")

        # --- YOLO OBJECT DETECTION (phone / drinking) ---
        # frame_b64 is passed in — see Change 3 below
        if frame_b64:
            yolo = detect_objects_yolo(frame_b64)
            if yolo.get('phone'):
                reasons.append("📱 Phone Detected")
                heuristic_distracted = True
            if yolo.get('drinking'):
                # only flag drinking if mouth model also agrees
                if mouth_class == 'Wide Open':
                    reasons.append("🥤 Drinking Suspected")
                    heuristic_distracted = True


        # HEURISTIC FUSION ENGINE
        if head_class == 'Frontal' and eye_class == 'Front':
            # Rule 1: Frontal + forward gaze → always Attentive
            final_is_distracted = False
        elif head_class in DISTRACTED_HEAD:
            # Rule 2: Distracted head pose → always Distracted
            reasons.append(f"Looking {head_class}")
            heuristic_distracted = True
            final_is_distracted  = True
        else:
            # Rule 3: No strong physical signal → trust AI + fatigue
            final_is_distracted = ai_distracted or heuristic_distracted

    else:
        head_class = eye_class = mouth_class = 'N/A'
        reasons             = []
        final_is_distracted = ai_distracted

    ms_info = {
        'face_detected':    face_found,
        'head':             head_class,
        'eye':              eye_class,
        'mouth':            mouth_class,
        'reasons':          reasons,
        'final_distracted': final_is_distracted
    }
    return ms_info, eye_closed_counter, fatigue_events_count, fatigue_max_duration, phone_suspect_counter

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

    face_crop, _ = get_face_crop(frame)
    img_input    = preprocess(face_crop if face_crop is not None else frame)
    prob         = float(model.predict(img_input, verbose=0)[0][0])
    risk         = get_risk_level(prob * 100)

    # ms_info, _, _, _ = get_multistream_info(frame, prob > 0.5, 0, 0, 0)
    final_label = "Distracted" if ms_info['final_distracted'] else "Attentive"

    gradcam_data = None
    try:
        heatmap      = generate_gradcam(img_input, model, last_conv_layer)
        overlay      = apply_gradcam_overlay(frame, heatmap)
        gradcam_data = {
            'original':   frame_to_base64(frame),
            'gradcam':    frame_to_base64(overlay),
            'confidence': round(prob * 100, 1)
        }
    except Exception as e:
        print(f"Grad-CAM error: {e}")

    return jsonify({
        'label':       final_label,
        'confidence':  round(prob * 100, 1),
        'risk_level':  risk,
        'multistream': ms_info,
        'gradcam':     gradcam_data
    })

# ================================
# LIVE SESSION MANAGEMENT
# ================================
@app.route('/live_start', methods=['POST'])
def live_start():
    """Call once when user opens the live webcam view."""
    session_id = str(uuid.uuid4())
    live_sessions[session_id] = new_session_state()
    print(f"✅ Live session started: {session_id}")
    return jsonify({'session_id': session_id})


@app.route('/live_end', methods=['POST'])
def live_end():
    """Call when user closes the live view. Returns final session summary."""
    session_id = request.json.get('session_id') if request.json else None
    if not session_id or session_id not in live_sessions:
        return jsonify({'error': 'Invalid or missing session_id'}), 400

    s = live_sessions.pop(session_id)
    total          = s['total_count']
    distracted_pct = round((s['distracted_count'] / total) * 100, 1) if total > 0 else 0
    safety_score, safety_grade, safety_message = calculate_safety_grade(
        distracted_pct, s['fatigue_events_count']
    )
    return jsonify({
        'summary': {
            'total_frames_analyzed': total,
            'distracted_pct':        distracted_pct,
            'attentive_pct':         round(100 - distracted_pct, 1),
            'risk_level':            get_risk_level(distracted_pct),
        },
        'safety': {
            'score':   safety_score,
            'grade':   safety_grade,
            'message': safety_message,
        },
        'fatigue': {
            'event_count':         s['fatigue_events_count'],
            'max_duration_frames': s['fatigue_max_duration'],
        }
    })


# ================================
# LIVE FRAME ANALYSIS
# Frontend sends JSON: { "session_id": "...", "image": "<base64>" }
# Returns instant per-frame result + running session stats
# ================================
@app.route('/analyze_live', methods=['POST'])
def analyze_live():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    session_id = data.get('session_id')
    image_b64  = data.get('image')

    if not session_id or session_id not in live_sessions:
        return jsonify({'error': 'Invalid or missing session_id. Call /live_start first.'}), 400
    if not image_b64:
        return jsonify({'error': 'No image data'}), 400

    try:
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Image decode failed: {e}'}), 400

    if frame is None:
        return jsonify({'error': 'Cannot read frame'}), 400

    s = live_sessions[session_id]

    # Main model — raised threshold (0.75) to counter domain gap on webcam
    face_crop, _ = get_face_crop(frame)
    img_input    = preprocess(face_crop if face_crop is not None else frame)
    prob         = float(model.predict(img_input, verbose=0)[0][0])
    ai_distracted = prob > 0.75

    # In analyze_live only — convert frame to b64 once and pass it:
    frame_b64_for_yolo = base64.b64encode(
        cv2.imencode('.jpg', frame)[1]
    ).decode('utf-8')


    # Multistream + Heuristic Fusion + Fatigue (live_mode=True)
    ms, s['eye_closed_counter'], s['fatigue_events_count'], s['fatigue_max_duration'], s['phone_suspect_counter'] = \
        get_multistream_info(
            frame, ai_distracted,
            s['eye_closed_counter'], s['fatigue_events_count'], s['fatigue_max_duration'],
            live_mode=True,
            frame_b64=frame_b64_for_yolo
        )

    # Temporal smoothing: 15-frame buffer, need 9/15 to confirm distracted
    s['temporal_buffer'].append(1 if ms['final_distracted'] else 0)
    smoothed_distracted = sum(s['temporal_buffer']) >= 9

    s['total_count'] += 1
    if smoothed_distracted:
        s['distracted_count'] += 1

    label          = "Distracted" if smoothed_distracted else "Attentive"
    distracted_pct = round((s['distracted_count'] / s['total_count']) * 100, 1)
    safety_score, safety_grade, safety_message = calculate_safety_grade(
        distracted_pct, s['fatigue_events_count']
    )

    return jsonify({
        'label':      label,
        'confidence': round(prob * 100, 1),
        'risk_level': get_risk_level(distracted_pct),
        'multistream': {
            'face_detected': ms['face_detected'],
            'head':          ms['head'],
            'eye':           ms['eye'],
            'mouth':         ms['mouth'],
            'reasons':       ms['reasons'],
        },
        'session': {
            'distracted_pct': distracted_pct,
            'fatigue_events': s['fatigue_events_count'],
            'safety_score':   safety_score,
            'safety_grade':   safety_grade,
            'safety_message': safety_message,
        }
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

    eye_closed_counter   = 0
    fatigue_events_count = 0
    fatigue_max_duration = 0

    temporal_buffer    = deque(maxlen=10)
    TEMPORAL_THRESHOLD = 6

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
    phone_suspect_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_every == 0:
            img_input     = preprocess(frame)
            prob          = float(model.predict(img_input, verbose=0)[0][0])
            ai_distracted = prob > 0.5  # Video keeps standard 0.5 threshold

            ms, eye_closed_counter, fatigue_events_count, fatigue_max_duration, phone_suspect_counter = \
                get_multistream_info(
                    frame, ai_distracted,
                    eye_closed_counter, fatigue_events_count, fatigue_max_duration,
                    phone_suspect_counter,
                    live_mode=False
                )

            temporal_buffer.append(1 if ms['final_distracted'] else 0)
            smoothed_distracted = sum(temporal_buffer) >= TEMPORAL_THRESHOLD
            label = "Distracted" if smoothed_distracted else "Attentive"

            confidences.append(round(prob, 4))
            labels.append(label)
            timestamps.append(round(frame_count / fps, 2) if fps > 0 else frame_count)
            frame_reasons.append(ms['reasons'])
            frame_head.append(ms['head'])
            frame_eye.append(ms['eye'])
            frame_mouth.append(ms['mouth'])

            color         = (0, 0, 255) if smoothed_distracted else (0, 255, 0)
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (10, 10), (550, 160), (0, 0, 0), -1)
            cv2.putText(display_frame, f"{label} ({prob * 100:.1f}%)",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if ms['face_detected']:
                cv2.putText(display_frame, f"Head:  {ms['head']}",
                            (20, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Eye:   {ms['eye']}",
                            (20, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Mouth: {ms['mouth']}",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if ms['reasons']:
                cv2.putText(display_frame, f"Reason: {', '.join(ms['reasons'])}",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            if len(processed_frames) < 50:
                processed_frames.append(frame_to_base64(display_frame))
            if prob > max_prob:
                max_prob = prob
                most_distracted_frame = frame.copy()

        frame_count += 1

    cap.release()
    os.unlink(tmp.name)

    total            = len(labels)
    distracted_count = labels.count("Distracted")
    attentive_count  = labels.count("Attentive")
    distracted_pct   = round((distracted_count / total) * 100, 1) if total > 0 else 0
    attentive_pct    = round((attentive_count  / total) * 100, 1) if total > 0 else 0
    risk_level       = get_risk_level(distracted_pct)

    safety_score, safety_grade, safety_message = calculate_safety_grade(
        distracted_pct, fatigue_events_count
    )

    alerts          = []
    in_alert        = False
    alert_start     = 0
    ALERT_THRESHOLD = 5
    consecutive     = 0

    for i, lbl in enumerate(labels):
        if lbl == "Distracted":
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

    gradcam_data = None
    if most_distracted_frame is not None:
        try:
            img_input       = preprocess(most_distracted_frame)
            heatmap         = generate_gradcam(img_input, model, last_conv_layer)
            gradcam_overlay = apply_gradcam_overlay(most_distracted_frame, heatmap)
            gradcam_data    = {
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
            'risk_level':            risk_level,
        },
        'safety': {
            'score':   safety_score,
            'grade':   safety_grade,
            'message': safety_message,
        },
        'fatigue': {
            'event_count':         fatigue_events_count,
            'max_duration_frames': fatigue_max_duration,
        },
        'frame_data': {
            'timestamps':  timestamps,
            'confidences': confidences,
            'labels':      labels,
        },
        'multistream': {
            'reasons': frame_reasons,
            'head':    frame_head,
            'eye':     frame_eye,
            'mouth':   frame_mouth,
        },
        'processed_frames': processed_frames,
        'alerts':    alerts,
        'gradcam':   gradcam_data,
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)