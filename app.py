# app.py - UPDATED with PDF report generation, completed_sessions fix, fatigue threshold fix
from flask import Flask, request, jsonify, send_file
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
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import io

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

# ================================
# LOAD CUSTOM MODEL (For live ensemble)
# ================================
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "video_model_custom_final.h5")
custom_model = None

if os.path.exists(CUSTOM_MODEL_PATH):
    try:
        custom_model = tf.keras.models.load_model(
            CUSTOM_MODEL_PATH,
            custom_objects={'loss': focal_loss()},
            compile=False
        )
        print("✅ Custom model loaded for live ensemble!")
    except Exception as e:
        print(f"⚠️ Custom model failed to load: {e}")
        custom_model = None
else:
    print("⚠️ Custom model not found — live mode will use SDDD model only")

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
# SESSION STORES
# ================================
# FATIGUE_THRESHOLD: at 1fps live capture, 3 = ~3 consecutive seconds of closed eyes
FATIGUE_THRESHOLD = 3

live_sessions      = {}   # active sessions being analyzed
completed_sessions = {}   # sessions saved after live_end — used for PDF download

def new_session_state():
    return {
        'eye_closed_counter':    0,
        'fatigue_events_count':  0,
        'fatigue_max_duration':  0,
        'temporal_buffer':       deque(maxlen=15),
        'distracted_count':      0,
        'total_count':           0,
        'phone_suspect_counter': 0,
        'max_prob':              0,
        'most_distracted_frame': None,
        'start_time':            datetime.now(),
    }

# ================================
# RECORDING STATE
# ================================
recording_sessions = {}  # session_id -> recording state

def start_recording(session_id):
    """Initialize recording frame buffer for a session."""
    if session_id in recording_sessions:
        print(f"⚠️ Recording already exists, skipping reinit: {session_id}")
        return
    recording_sessions[session_id] = {
        'frames':     [],
        'timestamps': [],
        'labels':     [],
        'reasons':    [],
        'fps':        6
    }
    print(f"🎬 Recording buffer initialized: {session_id}")

def stop_recording(session_id):
    """Write buffered frames to an MP4 file and return (path, rec_dict)."""
    if session_id not in recording_sessions:
        print(f"❌ No recording buffer for session: {session_id}")
        return None

    rec = recording_sessions.pop(session_id)

    if not rec['frames']:
        print(f"❌ No frames to write for session: {session_id}")
        return None

    h, w = rec['frames'][0].shape[:2]
    out_path = os.path.join(BASE_DIR, f"demo_{session_id[:8]}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(out_path, fourcc, rec['fps'], (w, h))

    if not out.isOpened():
        print(f"❌ Failed to open VideoWriter for {out_path}")
        return None

    for f in rec['frames']:
        out.write(f)
    out.release()
    print(f"✅ Video written: {out_path} ({len(rec['frames'])} frames)")
    return out_path, rec

# ================================
# ENSEMBLE PREDICTION
# ================================
def get_ensemble_prediction(frame, use_ensemble=True):
    img_input = preprocess(frame)
    prob_sddd = float(model.predict(img_input, verbose=0)[0][0])
    if not use_ensemble or custom_model is None:
        return prob_sddd
    prob_custom    = float(custom_model.predict(img_input, verbose=0)[0][0])
    ensemble_prob  = 0.60 * prob_sddd + 0.40 * prob_custom
    return ensemble_prob

# ================================
# PDF REPORT GENERATION
# ================================
def generate_session_report_pdf(session_data):
    """Generate a professional PDF report from session data."""
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter,
                                topMargin=0.5*inch, bottomMargin=0.5*inch)
    story  = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
        fontSize=24, textColor=colors.HexColor('#ff6600'),
        spaceAfter=6, fontName='Helvetica-Bold')
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor('#222222'),
        spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#444444'), spaceAfter=6)

    story.append(Paragraph("ADAS Driver Safety Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 0.2*inch))

    # Summary
    story.append(Paragraph("Session Summary", heading_style))
    sd = session_data['summary']
    summary_table = Table([
        ['Total Frames Analyzed', str(sd['total_frames_analyzed'])],
        ['Attentive Percentage',  f"{sd['attentive_pct']}%"],
        ['Distracted Percentage', f"{sd['distracted_pct']}%"],
        ['Risk Level',            sd['risk_level']],
    ], colWidths=[3*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR',  (0,0), (-1,-1), colors.black),
        ('ALIGN',      (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME',   (0,0), (0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('GRID',       (0,0), (-1,-1), 1, colors.HexColor('#e8e8e8')),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.2*inch))

    # Safety
    story.append(Paragraph("Safety Assessment", heading_style))
    sf = session_data['safety']
    grade_table = Table([
        ['Safety Grade', sf['grade']],
        ['Safety Score', f"{sf['score']} / 100"],
        ['Message',      sf['message']],
    ], colWidths=[3*inch, 2.5*inch])
    grade_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR',  (0,0), (-1,-1), colors.black),
        ('ALIGN',      (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME',   (0,0), (0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('GRID',       (0,0), (-1,-1), 1, colors.HexColor('#e8e8e8')),
    ]))
    story.append(grade_table)
    story.append(Spacer(1, 0.2*inch))

    # Fatigue
    story.append(Paragraph("Fatigue & Attention", heading_style))
    ft = session_data['fatigue']
    fatigue_table = Table([
        ['Fatigue Events Detected', str(ft['event_count'])],
        ['Max Duration (frames)',   str(ft['max_duration_frames'])],
    ], colWidths=[3*inch, 2.5*inch])
    fatigue_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR',  (0,0), (-1,-1), colors.black),
        ('ALIGN',      (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME',   (0,0), (0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('GRID',       (0,0), (-1,-1), 1, colors.HexColor('#e8e8e8')),
    ]))
    story.append(fatigue_table)
    story.append(Spacer(1, 0.2*inch))

    # Grad-CAM
    if session_data.get('gradcam'):
        story.append(Paragraph("Most Distracted Moment (Explainability)", heading_style))
        gradcam = session_data['gradcam']
        story.append(Paragraph(f"Confidence: {gradcam['confidence']}%", body_style))
        try:
            original_bytes = base64.b64decode(gradcam['original'])
            gradcam_bytes  = base64.b64decode(gradcam['gradcam'])
            original_img   = RLImage(io.BytesIO(original_bytes), width=2.5*inch, height=2*inch)
            gradcam_img    = RLImage(io.BytesIO(gradcam_bytes),  width=2.5*inch, height=2*inch)
            img_table      = Table([[original_img, gradcam_img]], colWidths=[2.8*inch, 2.8*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Original Frame vs Attention Heatmap (Grad-CAM)", body_style))
        except Exception as e:
            story.append(Paragraph(f"Could not include images: {e}", body_style))
        story.append(Spacer(1, 0.2*inch))

    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "Generated by ADAS Driver Distraction Detection System. "
        "MobileNetV2 + NST Augmentation | TensorFlow 2.13",
        ParagraphStyle('Footer', parent=styles['Normal'],
                       fontSize=8, textColor=colors.HexColor('#aaaaaa'))
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ================================
# HELPERS
# ================================
def generate_gradcam(img_array, mdl, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=mdl.inputs,
        outputs=[mdl.get_layer(last_conv_layer_name).output, mdl.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads       = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
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
    if distracted_pct < 20:  return "LOW"
    if distracted_pct < 50:  return "MEDIUM"
    return "HIGH"

YOLO_API = "http://localhost:5001"

def detect_objects_yolo(frame_b64):
    try:
        res = http_requests.post(f"{YOLO_API}/detect",
                                  json={'image': frame_b64}, timeout=0.8)
        if res.status_code == 200:
            return res.json()
        return {'phone': False, 'drinking': False}
    except Exception:
        return {'phone': False, 'drinking': False}

def get_ear_mar(frame_b64):
    try:
        res = requests.post("http://127.0.0.1:5002/analyze",
                            json={"image": frame_b64}, timeout=0.3)
        return res.json()
    except Exception:
        return {'ear': None, 'mar': None}

def get_face_crop(frame):
    h, w = frame.shape[:2]
    net  = _get_dnn_net()
    if net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        net.setInput(blob)
        detections = net.forward()
        best_conf, best_box = 0, None
        for i in range(detections.shape[2]):
            conf = float(detections[0,0,i,2])
            if conf < 0.5: continue
            if conf > best_conf:
                best_conf = conf
                best_box  = (
                    int(detections[0,0,i,3]*w), int(detections[0,0,i,4]*h),
                    int(detections[0,0,i,5]*w), int(detections[0,0,i,6]*h),
                )
        if best_box is not None:
            x1,y1,x2,y2 = best_box
            pad = 20
            x1=max(0,x1-pad); y1=max(0,y1-pad)
            x2=min(w,x2+pad); y2=min(h,y2+pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                return crop, (x1,y1,x2-x1,y2-y1)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x,y,fw,fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        return frame[y:y+fh, x:x+fw], (x,y,fw,fh)
    return None, None

def calculate_safety_grade(distracted_pct, fatigue_events):
    score = max(0, round(100 - distracted_pct - (fatigue_events * 15), 1))
    if score >= 90: return score, "A", "Excellent — Focused"
    if score >= 75: return score, "B", "Good — Minor Gaps"
    if score >= 60: return score, "C", "Caution — Frequent Distraction"
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
            eye_closed_counter, fatigue_events_count, fatigue_max_duration, phone_suspect_counter
        )

    face_crop, _ = get_face_crop(frame)
    if face_crop is None:
        return (
            {'face_detected': False, 'head': 'N/A', 'eye': 'N/A', 'mouth': 'N/A',
             'reasons': [], 'final_distracted': ai_distracted},
            eye_closed_counter, fatigue_events_count, fatigue_max_duration, phone_suspect_counter
        )

    face_img    = preprocess(face_crop)
    h_preds     = head_model.predict(face_img, verbose=0)[0]
    e_preds     = eye_model.predict(face_img,  verbose=0)[0]
    m_preds     = mouth_model.predict(face_img, verbose=0)[0]

    h_conf      = np.max(h_preds)
    head_class  = HEAD_CLASSES[np.argmax(h_preds)] if h_conf >= 0.75 else 'Frontal'
    eye_class   = EYE_CLASSES[np.argmax(e_preds)]
    eye_conf    = np.max(e_preds)
    mouth_class = MOUTH_CLASSES[np.argmax(m_preds)]

    reasons = []

    # EAR / MAR
    ear, mar = None, None
    if frame_b64:
        metrics = get_ear_mar(frame_b64)
        ear = metrics.get('ear')
        mar = metrics.get('mar')

    # ── Fatigue (eye close counter) ────────────────────────────────
    eye_conf_threshold = 0.65 if live_mode else 0.70
    eyes_closed = False
    if ear is not None and ear < 0.25:
        eyes_closed = True
    if eye_class == 'Closed' and eye_conf > eye_conf_threshold:
        eyes_closed = True

    is_yawning = mar is not None and mar > 0.6

    if eyes_closed and not is_yawning:
        eye_closed_counter += 1
        if eye_closed_counter == FATIGUE_THRESHOLD:
            fatigue_events_count += 1
            if "Fatigue Detected" not in reasons:
                reasons.append("Fatigue Detected")
        if eye_closed_counter >= FATIGUE_THRESHOLD:
            fatigue_max_duration = max(fatigue_max_duration, eye_closed_counter)
        if live_mode and eye_closed_counter >= 2:
            src = "(EAR)" if ear is not None else "(Model)"
            tag = f"Eyes Closed {src}"
            if tag not in reasons:
                reasons.append(tag)
    else:
        eye_closed_counter = 0

    # ── Yawning ────────────────────────────────────────────────────
    if is_yawning and "Yawning Detected" not in reasons:
        reasons.append("Yawning Detected")

    # ── Head pose — always add reason tag so frontend can show it ──
    if head_class in DISTRACTED_HEAD:
        tag = f"Looking {head_class}"
        if tag not in reasons:
            reasons.append(tag)

    # ── Phone usage heuristic ──────────────────────────────────────
    if head_class in ['Left', 'Right'] and eye_class in ['Left', 'Right', 'Front']:
        phone_suspect_counter += 1
        if phone_suspect_counter >= 5 and "Phone Usage Suspected" not in reasons:
            reasons.append("Phone Usage Suspected")
    else:
        phone_suspect_counter = 0

    # ── Drinking ───────────────────────────────────────────────────
    if head_class == 'Up' and mouth_class == 'Wide Open':
        if "Drinking Suspected" not in reasons:
            reasons.append("Drinking Suspected")

    # ── Eating ─────────────────────────────────────────────────────
    if mouth_class == 'Wide Open' and head_class in ['Frontal', 'Down']:
        if mar is not None and 0.40 < mar <= 0.60 and not is_yawning:
            if "Eating Suspected" not in reasons:
                reasons.append("Eating Suspected")

    # ── YOLO object detection ──────────────────────────────────────
    phone_detected    = False
    drinking_detected = False
    is_drinking_pose  = (
        head_class in ['Up', 'Diagonal Up Left', 'Diagonal Up Right'] and
        mouth_class == 'Wide Open' and
        mar is not None and mar > 0.40
    )

    if frame_b64:
        yolo = detect_objects_yolo(frame_b64)
        if yolo.get('phone'):
            phone_detected = True
            if "Phone Detected" not in reasons:
                reasons.append("Phone Detected")
        if yolo.get('drinking') and mouth_class == 'Wide Open':
            drinking_detected = True
            if "Drinking Detected" not in reasons:
                reasons.append("Drinking Detected")
        if is_drinking_pose and "Drinking Suspected (Pose)" not in reasons:
            reasons.append("Drinking Suspected (Pose)")

    # ── Fusion engine ──────────────────────────────────────────────
    if phone_detected:
        final_is_distracted = True
    elif drinking_detected or is_drinking_pose:
        final_is_distracted = True
    elif is_yawning:
        final_is_distracted = True
    elif "Fatigue Detected" in reasons or (live_mode and eye_closed_counter >= FATIGUE_THRESHOLD):
        final_is_distracted = True
    elif head_class in DISTRACTED_HEAD:
        # Head turned away — always distracted (reason tag already added above)
        final_is_distracted = True
    elif head_class == 'Frontal' and eye_class in ['Front', 'Up']:
        final_is_distracted = False
    else:
        final_is_distracted = ai_distracted

    ms_info = {
        'face_detected':    True,
        'head':             head_class,
        'eye':              eye_class,
        'mouth':            mouth_class,
        'reasons':          reasons,
        'final_distracted': final_is_distracted,
        'ear':              ear,
        'mar':              mar,
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

    ms_info, _, _, _, _ = get_multistream_info(frame, prob > 0.5, 0, 0, 0, 0)
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
    session_id = str(uuid.uuid4())
    live_sessions[session_id] = new_session_state()
    print(f"✅ Live session started: {session_id}")
    return jsonify({'session_id': session_id})


@app.route('/live_end', methods=['POST'])
def live_end():
    session_id = request.json.get('session_id') if request.json else None
    if not session_id or session_id not in live_sessions:
        return jsonify({'error': 'session_ended', 'stop': True}), 200

    s              = live_sessions.pop(session_id)
    total          = s['total_count']
    distracted_pct = round((s['distracted_count'] / total) * 100, 1) if total > 0 else 0
    safety_score, safety_grade, safety_message = calculate_safety_grade(
        distracted_pct, s['fatigue_events_count']
    )

    # Grad-CAM on most distracted frame
    gradcam_data = None
    if s['most_distracted_frame'] is not None:
        try:
            img_input = preprocess(s['most_distracted_frame'])
            heatmap   = generate_gradcam(img_input, model, last_conv_layer)
            overlay   = apply_gradcam_overlay(s['most_distracted_frame'], heatmap)
            gradcam_data = {
                'original':   frame_to_base64(s['most_distracted_frame']),
                'gradcam':    frame_to_base64(overlay),
                'confidence': round(s['max_prob'] * 100, 1)
            }
        except Exception as e:
            print(f"Live Grad-CAM error: {e}")

    final_data = {
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
        },
        'gradcam': gradcam_data
    }

    # ← KEY FIX: save completed session so /download_report works
    completed_sessions[session_id] = final_data

    return jsonify(final_data)


# ================================
# LIVE FRAME ANALYSIS
# ================================
@app.route('/analyze_live', methods=['POST'])
def analyze_live():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    session_id = data.get('session_id')
    image_b64  = data.get('image')

    if not session_id or session_id not in live_sessions:
        return jsonify({'error': 'session_ended', 'stop': True}), 200
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

    prob          = get_ensemble_prediction(frame, use_ensemble=True)
    ai_distracted = prob > 0.65

    if prob > s['max_prob']:
        s['max_prob']              = prob
        s['most_distracted_frame'] = frame.copy()

    frame_b64_for_services = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')

    ms, s['eye_closed_counter'], s['fatigue_events_count'], s['fatigue_max_duration'], s['phone_suspect_counter'] = \
        get_multistream_info(
            frame, ai_distracted,
            s['eye_closed_counter'], s['fatigue_events_count'],
            s['fatigue_max_duration'], s['phone_suspect_counter'],
            live_mode=True, frame_b64=frame_b64_for_services
        )

    # Temporal smoothing
    buffer_value = 1 if ms['final_distracted'] else 0
    s['temporal_buffer'].append(buffer_value)
    buffer_list      = list(s['temporal_buffer'])
    total_distracted = sum(buffer_list)
    recent_5         = buffer_list[-5:] if len(buffer_list) >= 5 else buffer_list
    recent_distracted = sum(recent_5)

    clear_attention = (
        ms['head'] == 'Frontal' and
        ms['eye'] in ['Front', 'Up'] and
        not ms['final_distracted'] and
        recent_distracted == 0
    )

    if clear_attention:
        smoothed_distracted = False
    elif total_distracted >= 12:
        smoothed_distracted = True
    elif total_distracted >= 9:
        smoothed_distracted = (recent_distracted >= 3)
    else:
        smoothed_distracted = False

    s['total_count'] += 1
    if smoothed_distracted:
        s['distracted_count'] += 1

    label          = "Distracted" if smoothed_distracted else "Attentive"
    distracted_pct = round((s['distracted_count'] / s['total_count']) * 100, 1)
    safety_score, safety_grade, safety_message = calculate_safety_grade(
        distracted_pct, s['fatigue_events_count']
    )

    print(f"🔍 Buffer:{total_distracted}/15 Recent:{recent_distracted}/5 "
          f"Label:{label} Head:{ms['head']} Eye:{ms['eye']} Reasons:{ms['reasons']}")

    # Append frame to recording buffer if active
    if session_id in recording_sessions:
        annotated = frame.copy()
        color     = (0,0,255) if smoothed_distracted else (0,255,0)
        cv2.rectangle(annotated, (8,8), (500,155), (0,0,0), -1)
        cv2.putText(annotated, f"{label} ({round(prob*100,1)}%)",
                    (15,42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(annotated, f"Head: {ms['head']}  Eye: {ms['eye']}  Mouth: {ms['mouth']}",
                    (15,72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(annotated, f"Grade: {safety_grade}  Score: {safety_score}  Distracted: {distracted_pct}%",
                    (15,95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        if ms['reasons']:
            cv2.putText(annotated, f"! {', '.join(ms['reasons'])}",
                        (15,118), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,165,255), 1)
        frame_num = len(recording_sessions[session_id]['frames'])
        ts        = round(frame_num / recording_sessions[session_id]['fps'], 1)
        cv2.putText(annotated, f"{ts}s", (15,145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
        recording_sessions[session_id]['frames'].append(annotated)
        recording_sessions[session_id]['timestamps'].append(ts)
        recording_sessions[session_id]['labels'].append(label)
        recording_sessions[session_id]['reasons'].append(ms['reasons'])

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
            'ear':           ms.get('ear'),
            'mar':           ms.get('mar'),
        },
        'session': {
            'distracted_pct':    distracted_pct,
            'fatigue_events':    s['fatigue_events_count'],
            'eye_closed_frames': s['eye_closed_counter'],
            'safety_score':      safety_score,
            'safety_grade':      safety_grade,
            'safety_message':    safety_message,
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
    temporal_buffer      = deque(maxlen=10)
    TEMPORAL_THRESHOLD   = 6

    confidences, labels, timestamps = [], [], []
    frame_reasons, frame_head, frame_eye, frame_mouth = [], [], [], []
    processed_frames = []
    frame_count      = 0
    sample_every     = max(1, int(fps / 2))
    max_prob         = 0
    most_distracted_frame = None
    phone_suspect_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % sample_every == 0:
            _, buf              = cv2.imencode('.jpg', frame)
            frame_b64_for_srv   = base64.b64encode(buf).decode('utf-8')
            img_input           = preprocess(frame)
            prob                = float(model.predict(img_input, verbose=0)[0][0])
            ai_distracted       = prob > 0.5

            ms, eye_closed_counter, fatigue_events_count, fatigue_max_duration, phone_suspect_counter = \
                get_multistream_info(
                    frame, ai_distracted,
                    eye_closed_counter, fatigue_events_count,
                    fatigue_max_duration, phone_suspect_counter,
                    live_mode=False, frame_b64=frame_b64_for_srv
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

            color         = (0,0,255) if smoothed_distracted else (0,255,0)
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (10,10), (550,160), (0,0,0), -1)
            cv2.putText(display_frame, f"{label} ({prob*100:.1f}%)",
                        (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if ms['face_detected']:
                cv2.putText(display_frame, f"Head:  {ms['head']}",
                            (20,75),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(display_frame, f"Eye:   {ms['eye']}",
                            (20,95),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(display_frame, f"Mouth: {ms['mouth']}",
                            (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if ms['reasons']:
                cv2.putText(display_frame, f"Reason: {', '.join(ms['reasons'])}",
                            (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

            if len(processed_frames) < 50:
                processed_frames.append(frame_to_base64(display_frame))
            if prob > max_prob:
                max_prob              = prob
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
    safety_score, safety_grade, safety_message = calculate_safety_grade(distracted_pct, fatigue_events_count)

    alerts, in_alert, alert_start, consecutive = [], False, 0, 0
    ALERT_THRESHOLD = 5
    for i, lbl in enumerate(labels):
        if lbl == "Distracted":
            consecutive += 1
            if consecutive >= ALERT_THRESHOLD and not in_alert:
                in_alert    = True
                alert_start = timestamps[i - ALERT_THRESHOLD + 1]
        else:
            if in_alert:
                alerts.append({'start': alert_start, 'end': timestamps[i-1]})
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
        'safety':  {'score': safety_score, 'grade': safety_grade, 'message': safety_message},
        'fatigue': {'event_count': fatigue_events_count, 'max_duration_frames': fatigue_max_duration},
        'frame_data': {'timestamps': timestamps, 'confidences': confidences, 'labels': labels},
        'multistream': {'reasons': frame_reasons, 'head': frame_head,
                        'eye': frame_eye, 'mouth': frame_mouth},
        'processed_frames': processed_frames,
        'alerts':  alerts,
        'gradcam': gradcam_data,
    })

# ================================
# RECORDING ENDPOINTS
# ================================
@app.route('/recording_start', methods=['POST'])
def recording_start():
    session_id = request.json.get('session_id')
    if not session_id or session_id not in live_sessions:
        print(f"❌ Invalid session for recording: {session_id}")
        return jsonify({'error': 'invalid session'}), 400
    if session_id in recording_sessions:
        return jsonify({'status': 'recording already started'}), 200
    start_recording(session_id)
    return jsonify({'status': 'recording started'})


@app.route('/recording_stop', methods=['POST'])
def recording_stop():
    session_id = request.json.get('session_id')
    if session_id not in recording_sessions:
        return jsonify({'error': 'no recording started'}), 400
    result = stop_recording(session_id)
    if result is None:
        return jsonify({'error': 'no frames recorded'}), 400
    out_path, rec = result
    total            = len(rec['labels'])
    distracted_count = rec['labels'].count('Distracted')
    distracted_pct   = round(distracted_count / total * 100, 1) if total > 0 else 0
    return jsonify({
        'video_path':      out_path,
        'video_filename':  os.path.basename(out_path),
        'total_frames':    total,
        'distracted_pct':  distracted_pct,
        'duration_seconds': round(total / rec['fps'], 1)
    })

# ================================
# DOWNLOAD ENDPOINTS
# ================================
@app.route('/download_video/<filename>', methods=['GET'])
def download_video(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'file not found'}), 404
    return send_file(path, mimetype='video/mp4', as_attachment=True, download_name=filename)


@app.route('/download_report/<session_id>', methods=['GET'])
def download_report(session_id):
    """Download the PDF report for a completed session."""
    if session_id not in completed_sessions:
        return jsonify({'error': 'session not found — make sure live_end was called first'}), 404

    pdf_buffer = generate_session_report_pdf(completed_sessions[session_id])
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'ADAS_Report_{session_id[:8]}.pdf'
    )

# ================================
# TRIM VIDEO
# ================================
@app.route('/trim_video', methods=['POST'])
def trim_video():
    data          = request.json
    filename      = data.get('filename')
    keep_segments = data.get('keep_segments', [])

    input_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'file not found'}), 404

    cap    = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 6
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = filename.replace('.mp4', '_trimmed.mp4')
    output_path     = os.path.join(BASE_DIR, output_filename)
    fourcc          = cv2.VideoWriter_fourcc(*'mp4v')
    out             = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        current_sec = frame_idx / fps
        for start, end in keep_segments:
            if start <= current_sec <= end:
                out.write(frame)
                break
        frame_idx += 1

    cap.release()
    out.release()

    return jsonify({
        'trimmed_filename': output_filename,
        'download_url':     f'/download_video/{output_filename}'
    })

# ================================
# HEALTH
# ================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)