# evaluate_fast.py  —  SELF-CONTAINED (no import from app.py)
#
# Root cause of "backend must be running" error:
#   The script did `from app import preprocess, get_multistream_info`.
#   app.py is a Flask server — importing it triggers model loading, Flask init,
#   and a circular import when app.py itself tries to import from app.py.
#   Fix: copy the 3 functions we need directly into this file. No Flask needed.
#
# All other fixes retained:
#   - shuffle=False on val_gen  (shuffle=True caused 95% → 52% accuracy collapse)
#   - Webcam class order check  (was causing AUC=0.016)
#   - Face crop before CNN      (matches app.py /analyze_live production path)
#   - Threshold 0.75 on webcam  (matches app.py /analyze_live)
#   - Correct face model paths  (v1=face_model_best.h5, v2=face_model_best_02.h5)
#   - Face model health check   (detects constant predictor, skips fusion)
#   - Ensemble support          (SDDD + custom model)

import os, json, warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import base64
import requests
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE        = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT"
DATASET     = os.path.join(BASE, "new_dataset")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT  = 0.2

# Model paths — edit these if your filenames differ
MAIN_MODEL_PATH   = os.path.join(BASE, "multistream", "multistream", "video_model_nst_v2_best.h5")
FACE_V1_PATH      = os.path.join(BASE, "face_model_best.h5")        # Haar-extracted
FACE_V2_PATH      = os.path.join(BASE, "face_model_best_02.h5")     # DNN-extracted
CUSTOM_MODEL_PATH = os.path.join(BASE, "video_model_custom_final.h5")

HEAD_MODEL_PATH  = os.path.join(BASE, "multistream", "multistream", "head_model_best.h5")
EYE_MODEL_PATH   = os.path.join(BASE, "multistream", "multistream", "eye_model_best.h5")
MOUTH_MODEL_PATH = os.path.join(BASE, "multistream", "multistream", "mouth_model_best.h5")

DNN_PROTO      = os.path.join(BASE, "deploy.prototxt")
DNN_CAFFEMODEL = os.path.join(BASE, "res10_300x300_ssd_iter_140000.caffemodel")

HEAD_CLASSES  = ['Diagonal Down Left', 'Diagonal Down Right', 'Diagonal Up Left',
                 'Diagonal Up Right', 'Down', 'Frontal', 'Left', 'Right', 'Up']
EYE_CLASSES   = ['Closed', 'Down', 'Front', 'Left', 'Right', 'Up']
MOUTH_CLASSES = ['Closed', 'Slight Open', 'Wide Open']
DISTRACTED_HEAD = ['Left', 'Right', 'Down', 'Diagonal Down Left',
                   'Diagonal Down Right', 'Diagonal Up Left', 'Diagonal Up Right']
FATIGUE_THRESHOLD = 12


# ─── FUNCTIONS COPIED FROM app.py (no Flask dependency) ──────────────────────

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

# Face detection — DNN with Haar fallback (identical logic to app.py)
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
_dnn_face_net = None

def _get_dnn_net():
    global _dnn_face_net
    if _dnn_face_net is None:
        if os.path.exists(DNN_PROTO) and os.path.exists(DNN_CAFFEMODEL):
            try:
                _dnn_face_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_CAFFEMODEL)
                print("✅ DNN face detector loaded")
            except Exception as e:
                print(f"⚠️  DNN face detector failed: {e}")
        else:
            print("⚠️  DNN files not found — using Haar fallback")
    return _dnn_face_net

def get_face_crop(frame):
    h, w = frame.shape[:2]
    net  = _get_dnn_net()
    if net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()
        best_conf, best_box = 0, None
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf >= 0.5 and conf > best_conf:
                best_conf = conf
                best_box  = (int(detections[0,0,i,3]*w), int(detections[0,0,i,4]*h),
                             int(detections[0,0,i,5]*w), int(detections[0,0,i,6]*h))
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            pad = 20
            x1 = max(0, x1-pad); y1 = max(0, y1-pad)
            x2 = min(w, x2+pad); y2 = min(h, y2+pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                return crop, (x1, y1, x2-x1, y2-y1)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, fw, fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        return frame[y:y+fh, x:x+fw], (x, y, fw, fh)
    return None, None

# Load multistream models once at script startup
_head_model = _eye_model = _mouth_model = None
_multistream_loaded = False
try:
    _head_model  = tf.keras.models.load_model(HEAD_MODEL_PATH,  compile=False)
    _eye_model   = tf.keras.models.load_model(EYE_MODEL_PATH,   compile=False)
    _mouth_model = tf.keras.models.load_model(MOUTH_MODEL_PATH, compile=False)
    _multistream_loaded = True
    print("✅ Multistream models loaded")
except Exception as e:
    print(f"⚠️  Multistream models not loaded: {e}")

def get_multistream_info(frame, ai_distracted,
                         eye_closed_counter, fatigue_events_count, fatigue_max_duration,
                         phone_suspect_counter=0, live_mode=False, frame_b64=None):
    """Exact copy of app.py get_multistream_info — runs offline, no Flask needed."""
    if not _multistream_loaded:
        return ({'face_detected': False, 'head': 'N/A', 'eye': 'N/A', 'mouth': 'N/A',
                 'reasons': [], 'final_distracted': ai_distracted},
                eye_closed_counter, fatigue_events_count, fatigue_max_duration, phone_suspect_counter)

    face_crop, _ = get_face_crop(frame)
    if face_crop is None:
        return ({'face_detected': False, 'head': 'N/A', 'eye': 'N/A', 'mouth': 'N/A',
                 'reasons': [], 'final_distracted': ai_distracted},
                eye_closed_counter, fatigue_events_count, fatigue_max_duration, phone_suspect_counter)

    face_img    = preprocess(face_crop)
    h_preds     = _head_model.predict(face_img,  verbose=0)[0]
    e_preds     = _eye_model.predict(face_img,   verbose=0)[0]
    m_preds     = _mouth_model.predict(face_img, verbose=0)[0]

    h_conf      = np.max(h_preds)
    head_class  = HEAD_CLASSES[np.argmax(h_preds)] if h_conf >= 0.75 else 'Frontal'
    eye_class   = EYE_CLASSES[np.argmax(e_preds)]
    eye_conf    = np.max(e_preds)
    mouth_class = MOUTH_CLASSES[np.argmax(m_preds)]

    reasons = []
    heuristic_distracted = False

    # EAR/MAR — silently skip if MediaPipe service not running (fine for offline eval)
    ear = mar = None
    if frame_b64:
        try:
            res = requests.post("http://127.0.0.1:5002/analyze",
                                json={"image": frame_b64}, timeout=0.3)
            if res.status_code == 200:
                d = res.json(); ear = d.get('ear'); mar = d.get('mar')
        except Exception:
            pass

    # Fatigue detection
    eye_conf_threshold = 0.65 if live_mode else 0.70
    eyes_closed = (ear is not None and ear < 0.25) or \
                  (eye_class == 'Closed' and eye_conf > eye_conf_threshold)

    if eyes_closed:
        eye_closed_counter += 1
        if eye_closed_counter == FATIGUE_THRESHOLD:
            fatigue_events_count += 1
            reasons.append("Fatigue Detected")
        if eye_closed_counter >= FATIGUE_THRESHOLD:
            heuristic_distracted = True
            fatigue_max_duration = max(fatigue_max_duration, eye_closed_counter)
        if live_mode and eye_closed_counter >= 3:
            heuristic_distracted = True
            if "Eyes Closed - Fatigue/Distraction" not in reasons:
                reasons.append(f"Eyes Closed {'(EAR)' if ear is not None else '(Model)'}")
    else:
        eye_closed_counter = 0

    if mar is not None and mar > 0.6:
        reasons.append("Yawning Detected"); heuristic_distracted = True

    if head_class in ['Left', 'Right'] and eye_class in ['Left', 'Right', 'Front']:
        phone_suspect_counter += 1
        if phone_suspect_counter >= 5:
            reasons.append("Phone Usage Suspected"); heuristic_distracted = True
    else:
        phone_suspect_counter = 0

    if head_class == 'Down' and mouth_class == 'Wide Open':
        reasons.append("Drinking Suspected"); heuristic_distracted = True

    if mouth_class == 'Wide Open' and head_class in ['Frontal', 'Down']:
        if mar is None or mar > 0.35:
            reasons.append("Eating Suspected"); heuristic_distracted = True

    # YOLO — silently skip if service not running
    phone_detected = drinking_detected = False
    if frame_b64:
        try:
            res = requests.post("http://localhost:5001/detect",
                                json={'image': frame_b64}, timeout=0.8)
            if res.status_code == 200:
                yolo = res.json()
                if yolo.get('phone'):
                    phone_detected = True; reasons.append("Phone Detected"); heuristic_distracted = True
                if yolo.get('drinking') and mouth_class == 'Wide Open':
                    drinking_detected = True; reasons.append("Drinking Detected"); heuristic_distracted = True
        except Exception:
            pass

    # Fusion engine (identical to app.py)
    if phone_detected or drinking_detected:
        final_is_distracted = True
    elif head_class == 'Frontal' and eye_class == 'Front':
        final_is_distracted = False
    elif head_class in DISTRACTED_HEAD:
        reasons.append(f"Looking {head_class}"); heuristic_distracted = True
        final_is_distracted = True
    else:
        final_is_distracted = ai_distracted or heuristic_distracted

    return ({'face_detected': True, 'head': head_class, 'eye': eye_class,
             'mouth': mouth_class, 'reasons': reasons,
             'final_distracted': final_is_distracted, 'ear': ear, 'mar': mar},
            eye_closed_counter, fatigue_events_count, fatigue_max_duration, phone_suspect_counter)

# ─── END OF COPIED FUNCTIONS ──────────────────────────────────────────────────


# ─── VALIDATION GENERATOR ─────────────────────────────────────────────────────
# IMPORTANT: shuffle=False is required.
# ImageDataGenerator validation_split takes the last val_split% of files
# alphabetically. shuffle=True randomises the split, leaking train data into val
# and destroying all metrics (caused 95% -> 52% accuracy collapse previously).
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
val_gen = val_datagen.flow_from_directory(
    DATASET, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation',
    shuffle=False   # ← DO NOT CHANGE TO TRUE
)
TRUE_LABELS = val_gen.classes
TRAIN_CLASS_INDICES = val_gen.class_indices
print(f"Val set: {val_gen.samples} images | Classes: {TRAIN_CLASS_INDICES}\n")


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_preds(model, gen):
    gen.reset()
    return model.predict(gen, verbose=1).flatten()

def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = np.mean(y_true == y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    return dict(accuracy=round(acc*100,2), precision=round(p*100,2),
                recall=round(r*100,2), f1=round(f*100,2),
                auc=round(roc_auc,4), fpr=fpr, tpr=tpr)

def check_face_model_health(y_true, y_prob, name):
    unique_preds = np.unique((y_prob >= 0.5).astype(int))
    if len(unique_preds) == 1:
        print(f"\n⚠️  WARNING: '{name}' is a CONSTANT PREDICTOR "
              f"(always outputs class {unique_preds[0]}).")
        print("   Retrain with: class_weight={{0: 2.0, 1: 1.0}}\n")
        return False
    return True

def build_frozen(backbone_fn):
    base = backbone_fn(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inputs=base.input, outputs=out)
    m.compile(optimizer='adam', loss='binary_crossentropy')
    return m

def print_table(results, title="RESULTS"):
    print(f"\n{'='*85}")
    print(title)
    print(f"{'Model':<42} {'Acc%':>6} {'Prec%':>7} {'Rec%':>6} {'F1%':>6} {'AUC':>7}")
    print(f"{'-'*85}")
    for name, m in results.items():
        print(f"{name:<42} {m['accuracy']:>6} {m['precision']:>7} {m['recall']:>6} {m['f1']:>6} {m['auc']:>7}")
    print(f"{'='*85}\n")

def save_json(results, fname):
    clean = {k: {m: (v.tolist() if hasattr(v,'tolist') else v)
                 for m, v in vals.items() if m not in ['fpr','tpr']}
             for k, vals in results.items()}
    with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
        json.dump(clean, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════════
print("="*60)
print("EXPERIMENT 1: Model Comparison")
print("="*60)
results = {}
face_v2_healthy = False

print("\n[1/6] Your model (MobileNetV2 + NST + Focal Loss)...")
main_model = tf.keras.models.load_model(
    MAIN_MODEL_PATH, custom_objects={'loss': focal_loss()}, compile=False)
results['Ours: MobileNetV2 + NST + Focal Loss'] = metrics(TRUE_LABELS, get_preds(main_model, val_gen))
del main_model

print("\n[2/6] Face model v2 (DNN-extracted)...")
if os.path.exists(FACE_V2_PATH):
    face_v2 = tf.keras.models.load_model(
        FACE_V2_PATH, custom_objects={'loss': focal_loss()}, compile=False)
    face_preds_v2 = get_preds(face_v2, val_gen)
    face_v2_healthy = check_face_model_health(TRUE_LABELS, face_preds_v2, "Face model v2")
    results['Ours: Face crop model (DNN)'] = metrics(TRUE_LABELS, face_preds_v2)
    del face_v2
else:
    print(f"⚠️  Not found: {FACE_V2_PATH} — skipping")

print("\n[3/6] VGG16 frozen pretrained (zero-shot)...")
vgg = build_frozen(VGG16)
results['VGG16 (frozen ImageNet, zero-shot)'] = metrics(TRUE_LABELS, get_preds(vgg, val_gen))
del vgg

print("\n[4/6] ResNet50 frozen pretrained (zero-shot)...")
resnet = build_frozen(ResNet50)
results['ResNet50 (frozen ImageNet, zero-shot)'] = metrics(TRUE_LABELS, get_preds(resnet, val_gen))
del resnet

print("\n[5/6] MobileNetV2 frozen pretrained (zero-shot)...")
mob = build_frozen(MobileNetV2)
results['MobileNetV2 (frozen ImageNet, zero-shot)'] = metrics(TRUE_LABELS, get_preds(mob, val_gen))
del mob

print("\n[6/6] Custom model (Driver-specific training)...")
if os.path.exists(CUSTOM_MODEL_PATH):
    custom_m = tf.keras.models.load_model(
        CUSTOM_MODEL_PATH, custom_objects={'loss': focal_loss()}, compile=False)
    results['Ours: Custom-trained (Driver 1-4)'] = metrics(TRUE_LABELS, get_preds(custom_m, val_gen))
    del custom_m
else:
    print(f"⚠️  Custom model not found at {CUSTOM_MODEL_PATH} — skipping")

print_table(results, "EXPERIMENT 1: COMPARISON TABLE")
save_json(results, "comparison_results.json")


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: ABLATION STUDY
# ════════════════════════════════════════════════════════════════════════════════
print("="*60)
print("EXPERIMENT 2: Ablation Study")
print("="*60)
ablation = {}

print("\n[Ablation 1/6] MobileNetV2 zero-shot (no training)...")
mob_zero = build_frozen(MobileNetV2)
ablation['No training (zero-shot ImageNet)'] = metrics(TRUE_LABELS, get_preds(mob_zero, val_gen))
del mob_zero

# Face v1 — Haar extracted (DIFFERENT file from v2)
if os.path.exists(FACE_V1_PATH):
    print("\n[Ablation 2/6] Face model v1 (Haar cascade extracted)...")
    face_v1 = tf.keras.models.load_model(
        FACE_V1_PATH, custom_objects={'loss': focal_loss()}, compile=False)
    p1 = get_preds(face_v1, val_gen)
    check_face_model_health(TRUE_LABELS, p1, "Face model v1 (Haar)")
    ablation['MobileNetV2 trained, Haar face crops'] = metrics(TRUE_LABELS, p1)
    del face_v1
else:
    print(f"⚠️  Face v1 not found: {FACE_V1_PATH} — skipping")

# Face v2 — DNN extracted (DIFFERENT file from v1)
if os.path.exists(FACE_V2_PATH):
    print("\n[Ablation 3/6] Face model v2 (DNN extracted)...")
    face_v2 = tf.keras.models.load_model(
        FACE_V2_PATH, custom_objects={'loss': focal_loss()}, compile=False)
    preds_face_v2 = get_preds(face_v2, val_gen)
    check_face_model_health(TRUE_LABELS, preds_face_v2, "Face model v2 (DNN)")
    ablation['MobileNetV2 trained, DNN face crops'] = metrics(TRUE_LABELS, preds_face_v2)
    del face_v2

print("\n[Ablation 4/6] Main model (NST augmented)...")
main_model = tf.keras.models.load_model(
    MAIN_MODEL_PATH, custom_objects={'loss': focal_loss()}, compile=False)
preds_main = get_preds(main_model, val_gen)
ablation['MobileNetV2 + NST augmentation'] = metrics(TRUE_LABELS, preds_main)

print("\n[Ablation 5/6] Full fusion (0.7 main + 0.3 face)...")
if os.path.exists(FACE_V2_PATH):
    face_v2_fuse = tf.keras.models.load_model(
        FACE_V2_PATH, custom_objects={'loss': focal_loss()}, compile=False)
    preds_face_fuse = get_preds(face_v2_fuse, val_gen)
    face_fuse_healthy = check_face_model_health(TRUE_LABELS, preds_face_fuse, "Face model v2 (fusion)")
    if face_fuse_healthy:
        fused = 0.70 * preds_main + 0.30 * preds_face_fuse
        ablation['Full fusion (0.7 main + 0.3 face model)'] = metrics(TRUE_LABELS, fused)
        print("✅ Fusion computed.")
    else:
        print("⛔ Fusion SKIPPED — face model is a constant predictor.")
        ablation['Full fusion (SKIPPED — face model broken)'] = ablation['MobileNetV2 + NST augmentation']
    del face_v2_fuse
else:
    ablation['Full fusion (SKIPPED — face model missing)'] = ablation['MobileNetV2 + NST augmentation']

print("\n[Ablation 6/6] Ensemble (60% SDDD + 40% Custom)...")
if os.path.exists(CUSTOM_MODEL_PATH):
    custom_abl = tf.keras.models.load_model(
        CUSTOM_MODEL_PATH, custom_objects={'loss': focal_loss()}, compile=False)
    preds_custom_abl = get_preds(custom_abl, val_gen)
    ensemble_preds = 0.60 * preds_main + 0.40 * preds_custom_abl
    ablation['Ensemble (60% SDDD + 40% Custom)'] = metrics(TRUE_LABELS, ensemble_preds)
    del custom_abl
    print("✅ Ensemble evaluation complete.")
else:
    print(f"⚠️  Custom model not found — skipping ensemble ablation")

del main_model

print_table(ablation, "EXPERIMENT 2: ABLATION STUDY")
save_json(ablation, "ablation_results.json")


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: ROC CURVES
# ════════════════════════════════════════════════════════════════════════════════
print("Plotting ROC curves...")
fig, ax = plt.subplots(figsize=(9, 7))
colors = ['#1F4E79', '#2E75B6', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD']
for (name, m), color in zip(results.items(), colors):
    ax.plot(m['fpr'], m['tpr'],
            label=f"{name} (AUC={m['auc']:.3f})", color=color, linewidth=2)
ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — Model Comparison\n(Held-out validation set, 20% split)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(RESULTS_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150); plt.close()
print(f"ROC curve saved: {roc_path}")


# ════════════════════════════════════════════════════════════════════════════════
# ABLATION BAR CHART
# ════════════════════════════════════════════════════════════════════════════════
print("Plotting ablation chart...")
configs  = list(ablation.keys())
accs     = [ablation[c]['accuracy']  for c in configs]
f1s      = [ablation[c]['f1']        for c in configs]
aucs_pct = [ablation[c]['auc'] * 100 for c in configs]
x     = np.arange(len(configs))
width = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
b1 = ax.bar(x - width, accs,     width, label='Accuracy %', color='#1F4E79', alpha=0.85)
b2 = ax.bar(x,         f1s,      width, label='F1 Score %',  color='#2E75B6', alpha=0.85)
b3 = ax.bar(x + width, aucs_pct, width, label='AUC x100',    color='#70AD47', alpha=0.85)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Ablation Study — Contribution of Each Component', fontsize=14, fontweight='bold')
ax.set_xticks(x)
short_labels = [c.replace(' (', '\n(').replace(', ', ',\n') for c in configs]
ax.set_xticklabels(short_labels, fontsize=8, rotation=10, ha='right')
ax.legend(fontsize=10); ax.set_ylim(40, 110); ax.grid(axis='y', alpha=0.3)
for bars in [b1, b2, b3]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)
plt.tight_layout()
ablation_path = os.path.join(RESULTS_DIR, "ablation_study.png")
plt.savefig(ablation_path, dpi=150); plt.close()
print(f"Ablation chart saved: {ablation_path}")


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: CROSS-CAMERA (WEBCAM)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 4: Cross-Camera Generalisation")
print("="*60)

webcam_dir = os.path.join(BASE, "webcam_test_set")
att_dir    = os.path.join(webcam_dir, "attentive")
dis_dir    = os.path.join(webcam_dir, "distracted")
os.makedirs(att_dir, exist_ok=True)
os.makedirs(dis_dir, exist_ok=True)

existing_att = len([f for f in os.listdir(att_dir) if f.endswith('.jpg')])
existing_dis = len([f for f in os.listdir(dis_dir) if f.endswith('.jpg')])

if existing_att < 30 or existing_dis < 30:
    print(f"\nNeed webcam frames (have {existing_att} att, {existing_dis} dis — need 30+ each)")
    print("Opening webcam — press A=attentive, D=distracted, Q=done")
    cap = cv2.VideoCapture(0)
    count = {'attentive': existing_att, 'distracted': existing_dis}
    TARGET = 50
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.putText(frame,
            f"A=attentive({count['attentive']}/{TARGET})  D=distracted({count['distracted']}/{TARGET})  Q=quit",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("Webcam Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            cv2.imwrite(os.path.join(att_dir, f"att_{count['attentive']:04d}.jpg"), frame)
            count['attentive'] += 1
        elif key == ord('d'):
            cv2.imwrite(os.path.join(dis_dir, f"dis_{count['distracted']:04d}.jpg"), frame)
            count['distracted'] += 1
        elif key == ord('q'): break
        if count['attentive'] >= TARGET and count['distracted'] >= TARGET: break
    cap.release(); cv2.destroyAllWindows()

# Class order verification
webcam_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    webcam_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)
webcam_labels_raw    = webcam_gen.classes
WEBCAM_CLASS_INDICES = webcam_gen.class_indices

print(f"\nWebcam test set: {webcam_gen.samples} images")
print(f"Training class indices : {TRAIN_CLASS_INDICES}")
print(f"Webcam   class indices : {WEBCAM_CLASS_INDICES}")

classes_match = (TRAIN_CLASS_INDICES == WEBCAM_CLASS_INDICES)
if not classes_match:
    print("⚠️  CLASS ORDER MISMATCH — flipping webcam labels.")
    webcam_labels = 1 - webcam_labels_raw
else:
    print("✅ Class order matches.")
    webcam_labels = webcam_labels_raw

cross = {}

# Load models for webcam evaluation
main_model = tf.keras.models.load_model(
    MAIN_MODEL_PATH, custom_objects={'loss': focal_loss()}, compile=False)

custom_wcam = None
if os.path.exists(CUSTOM_MODEL_PATH):
    custom_wcam = tf.keras.models.load_model(
        CUSTOM_MODEL_PATH, custom_objects={'loss': focal_loss()}, compile=False)
    print("✅ Custom model loaded for webcam ensemble")

print("\n[Hybrid Evaluation] Running CNN + heuristics on webcam set...")

y_true          = []
y_pred_sddd     = []
y_pred_ensemble = []
y_pred_hybrid   = []

WEBCAM_THRESHOLD   = 0.75   # Matches /analyze_live threshold in app.py
ENSEMBLE_THRESHOLD = 0.65   # Slightly lower — ensemble is more calibrated

for class_name in ['attentive', 'distracted']:
    class_dir    = os.path.join(webcam_dir, class_name)
    raw_label    = WEBCAM_CLASS_INDICES.get(class_name, -1)
    if raw_label == -1: continue
    target_label = (1 - raw_label) if not classes_match else raw_label

    img_list = [f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in img_list:
        frame = cv2.imread(os.path.join(class_dir, img_name))
        if frame is None: continue

        # Face crop first — matches /analyze_live production path
        face_crop, _ = get_face_crop(frame)
        cnn_input    = preprocess(face_crop if face_crop is not None else frame)

        prob_sddd = float(main_model.predict(cnn_input, verbose=0)[0][0])
        if custom_wcam is not None:
            prob_custom   = float(custom_wcam.predict(cnn_input, verbose=0)[0][0])
            prob_ensemble = 0.60 * prob_sddd + 0.40 * prob_custom
        else:
            prob_ensemble = prob_sddd

        aligned_sddd     = (1.0 - prob_sddd)     if not classes_match else prob_sddd
        aligned_ensemble = (1.0 - prob_ensemble) if not classes_match else prob_ensemble

        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        ms_info, _, _, _, _ = get_multistream_info(
            frame,
            ai_distracted=(aligned_ensemble > ENSEMBLE_THRESHOLD),
            eye_closed_counter=0, fatigue_events_count=0,
            fatigue_max_duration=0, phone_suspect_counter=0,
            live_mode=True, frame_b64=frame_b64
        )

        y_true.append(target_label)
        y_pred_sddd.append(aligned_sddd)
        y_pred_ensemble.append(aligned_ensemble)
        y_pred_hybrid.append(1.0 if ms_info.get('final_distracted', False) else 0.0)

y_true          = np.array(y_true)
y_pred_sddd     = np.array(y_pred_sddd)
y_pred_ensemble = np.array(y_pred_ensemble)
y_pred_hybrid   = np.array(y_pred_hybrid)

cross['CNN only (SDDD) — webcam']               = metrics(y_true, y_pred_sddd,     threshold=WEBCAM_THRESHOLD)
if custom_wcam is not None:
    cross['Ensemble (SDDD+Custom) — webcam']    = metrics(y_true, y_pred_ensemble, threshold=ENSEMBLE_THRESHOLD)
cross['Hybrid (Ensemble+Multistream) — webcam'] = metrics(y_true, y_pred_hybrid)
print("✅ Hybrid evaluation complete.")

# Face model on webcam
if os.path.exists(FACE_V2_PATH):
    face_wcam = tf.keras.models.load_model(
        FACE_V2_PATH, custom_objects={'loss': focal_loss()}, compile=False)
    webcam_gen.reset()
    raw_face_preds = get_preds(face_wcam, webcam_gen)
    aligned_face   = (1.0 - raw_face_preds) if not classes_match else raw_face_preds
    cross['Face model v2 — webcam'] = metrics(webcam_labels, aligned_face)
    del face_wcam

del main_model
if custom_wcam is not None: del custom_wcam

print_table(cross, "EXPERIMENT 4: CROSS-CAMERA RESULTS (trained SDDD → tested webcam)")

val_acc  = results.get('Ours: MobileNetV2 + NST + Focal Loss', {}).get('accuracy', 0)
wcam_acc = cross.get('CNN only (SDDD) — webcam', {}).get('accuracy', 0)
print(f"Domain gap (val → webcam): {val_acc:.2f}% → {wcam_acc:.2f}%  (Δ = {val_acc - wcam_acc:.2f}%)\n")

save_json(cross, "cross_camera_results.json")

# Cross-camera bar chart
try:
    with open(os.path.join(RESULTS_DIR, "comparison_results.json")) as f:
        comp = json.load(f)
    model_names = ['Ours (MobileNetV2+NST)', 'Ensemble (SDDD+Custom)', 'Face model v2']
    val_accs  = [comp.get('Ours: MobileNetV2 + NST + Focal Loss', {}).get('accuracy', 0),
                 ablation.get('Ensemble (60% SDDD + 40% Custom)', {}).get('accuracy', 0),
                 comp.get('Ours: Face crop model (DNN)', {}).get('accuracy', 0)]
    wcam_accs = [cross.get('CNN only (SDDD) — webcam', {}).get('accuracy', 0),
                 cross.get('Ensemble (SDDD+Custom) — webcam', {}).get('accuracy', 0),
                 cross.get('Face model v2 — webcam', {}).get('accuracy', 0)]
    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, val_accs,  width, label='Validation (SDDD)', color='#1F4E79', alpha=0.85)
    ax.bar(x + width/2, wcam_accs, width, label='Webcam (unseen)',    color='#E74C3C', alpha=0.85)
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Cross-Camera Generalisation', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=10); ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    cross_path = os.path.join(RESULTS_DIR, "cross_camera.png")
    plt.savefig(cross_path, dpi=150); plt.close()
    print(f"Cross-camera chart saved: {cross_path}")
except Exception as e:
    print(f"Cross-camera chart skipped: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: TEMPORAL EVALUATION (OPTIONAL)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 5: Temporal Behavior (Video Clips)")
print("="*60)

video_test_dir = os.path.join(BASE, "video_test_clips")
if os.path.exists(video_test_dir):
    main_model = tf.keras.models.load_model(
        MAIN_MODEL_PATH, custom_objects={'loss': focal_loss()}, compile=False)
    for video_file in os.listdir(video_test_dir):
        if not video_file.endswith(('.mp4', '.avi')): continue
        cap = cv2.VideoCapture(os.path.join(video_test_dir, video_file))
        true_label = 1 if 'distracted' in video_file.lower() else 0
        eye_closed_counter = fatigue_events = fatigue_max = phone_counter = 0
        frame_predictions = []; frame_count = 0
        while cap.isOpened() and frame_count < 150:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % 3 != 0:
                frame_count += 1; continue
            fc, _ = get_face_crop(frame)
            img_input = preprocess(fc if fc is not None else frame)
            raw_prob  = float(main_model.predict(img_input, verbose=0)[0][0])
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            ms_info, eye_closed_counter, fatigue_events, fatigue_max, phone_counter = \
                get_multistream_info(frame, (raw_prob > 0.75),
                    eye_closed_counter, fatigue_events, fatigue_max, phone_counter,
                    live_mode=True, frame_b64=frame_b64)
            frame_predictions.append(1.0 if ms_info['final_distracted'] else 0.0)
            frame_count += 1
        cap.release()
        avg = np.mean(frame_predictions) if frame_predictions else 0.5
        print(f"{video_file}: True={true_label}, Pred={1 if avg>0.5 else 0}, "
              f"Fatigue={fatigue_events}, Avg={avg:.2f}")
    del main_model
    print("Temporal evaluation complete.")
else:
    print("No video_test_clips/ directory — skipping")


# ════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ALL DONE — files saved to results/")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  {f}")
print("="*60)

print("\n── Key numbers ──")
main = results.get('Ours: MobileNetV2 + NST + Focal Loss', {})
wcam = cross.get('CNN only (SDDD) — webcam', {})
ens  = cross.get('Ensemble (SDDD+Custom) — webcam', {})
print(f"Main model  val accuracy : {main.get('accuracy','?')}%  AUC={main.get('auc','?')}")
print(f"Main model  webcam acc   : {wcam.get('accuracy','?')}%  AUC={wcam.get('auc','?')}")
if ens:
    print(f"Ensemble    webcam acc   : {ens.get('accuracy','?')}%  AUC={ens.get('auc','?')}")
face = results.get('Ours: Face crop model (DNN)', {})
print(f"Face model  val accuracy : {face.get('accuracy','?')}%  AUC={face.get('auc','?')}")
if not face_v2_healthy:
    print("  ↳ Face model needs retraining (constant predictor)")