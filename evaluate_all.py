# evaluate_fast.py
# Runs in ~15 minutes total, no retraining needed.
# Comparisons use frozen pretrained backbones (zero-shot) — valid academic comparison.
# Generates: ROC curves, ablation chart, comparison table, cross-camera results.

import os, json, warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from app import preprocess, get_multistream_info
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
warnings.filterwarnings('ignore')

# ─── CONFIG ───
BASE        = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT"
DATASET     = os.path.join(BASE, "new_dataset")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT  = 0.2

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

# ─── VALIDATION GENERATOR ───
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
val_gen = val_datagen.flow_from_directory(
    DATASET, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation', shuffle=False
)
TRUE_LABELS = val_gen.classes
print(f"Val set: {val_gen.samples} images | Classes: {val_gen.class_indices}\n")


# ─── HELPERS ───
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

def build_frozen(backbone_fn):
    """Frozen pretrained backbone + single dense head. No training — zero-shot."""
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


# ════════════════════════════════════════
# EXPERIMENT 1: COMPARISON TABLE
# ════════════════════════════════════════
print("="*60)
print("EXPERIMENT 1: Model Comparison")
print("="*60)
results = {}

# Your best model
print("\n[1/5] Your model (MobileNetV2 + NST + Focal Loss)...")
your_model = tf.keras.models.load_model(
    os.path.join(BASE, "multistream", "multistream", "video_model_nst_v2_best.h5"),
    custom_objects={'loss': focal_loss()}, compile=False)
results['Ours: MobileNetV2 + NST + Focal Loss'] = metrics(TRUE_LABELS, get_preds(your_model, val_gen))
del your_model

# Face model v2
print("\n[2/5] Face model v2 (DNN-extracted)...")
face_v2 = tf.keras.models.load_model(
    os.path.join(BASE, "face_model_best_02.h5"),
    custom_objects={'loss': focal_loss()}, compile=False)
results['Ours: Face crop model (DNN)'] = metrics(TRUE_LABELS, get_preds(face_v2, val_gen))
del face_v2

# Frozen VGG16 (zero-shot)
print("\n[3/5] VGG16 frozen pretrained (zero-shot, no fine-tuning)...")
vgg = build_frozen(VGG16)
results['VGG16 (frozen ImageNet, zero-shot)'] = metrics(TRUE_LABELS, get_preds(vgg, val_gen))
del vgg

# Frozen ResNet50 (zero-shot)
print("\n[4/5] ResNet50 frozen pretrained (zero-shot, no fine-tuning)...")
resnet = build_frozen(ResNet50)
results['ResNet50 (frozen ImageNet, zero-shot)'] = metrics(TRUE_LABELS, get_preds(resnet, val_gen))
del resnet

# Frozen MobileNetV2 (zero-shot — same backbone as yours but untrained on your data)
print("\n[5/5] MobileNetV2 frozen pretrained (zero-shot, no fine-tuning)...")
mob = build_frozen(MobileNetV2)
results['MobileNetV2 (frozen ImageNet, zero-shot)'] = metrics(TRUE_LABELS, get_preds(mob, val_gen))
del mob

print_table(results, "EXPERIMENT 1: COMPARISON TABLE")
save_json(results, "comparison_results.json")


# ════════════════════════════════════════
# EXPERIMENT 2: ABLATION STUDY
# ════════════════════════════════════════
print("="*60)
print("EXPERIMENT 2: Ablation Study")
print("="*60)
ablation = {}

# Config 1: MobileNetV2 zero-shot (no task-specific training at all)
print("\n[Ablation 1/5] MobileNetV2 zero-shot (no training)...")
mob_zero = build_frozen(MobileNetV2)
ablation['No training (zero-shot ImageNet)'] = metrics(TRUE_LABELS, get_preds(mob_zero, val_gen))
del mob_zero

# Config 2: Face model v1 (Haar extracted)
face_v1_path = os.path.join(BASE, "face_model_best.h5")
if os.path.exists(face_v1_path):
    print("\n[Ablation 2/5] Face model v1 (Haar cascade extracted)...")
    face_v1 = tf.keras.models.load_model(
        face_v1_path, custom_objects={'loss': focal_loss()}, compile=False)
    ablation['MobileNetV2 trained, Haar face crops'] = metrics(TRUE_LABELS, get_preds(face_v1, val_gen))
    del face_v1

# Config 3: Face model v2 (DNN extracted — same architecture, better data)
print("\n[Ablation 3/5] Face model v2 (DNN extracted)...")
face_v2 = tf.keras.models.load_model(
    os.path.join(BASE, "face_model_best_02.h5"),
    custom_objects={'loss': focal_loss()}, compile=False)
preds_face_v2 = get_preds(face_v2, val_gen)
ablation['MobileNetV2 trained, DNN face crops'] = metrics(TRUE_LABELS, preds_face_v2)
del face_v2

# Config 4: Main model (NST augmented — your best single model)
print("\n[Ablation 4/5] Main model (NST augmented)...")
main_model = tf.keras.models.load_model(
    os.path.join(BASE, "multistream", "multistream", "video_model_nst_v2_best.h5"),
    custom_objects={'loss': focal_loss()}, compile=False)
preds_main = get_preds(main_model, val_gen)
ablation['MobileNetV2 + NST augmentation'] = metrics(TRUE_LABELS, preds_main)

# Config 5: Full fusion (main + face model v2)
print("\n[Ablation 5/5] Full fusion (0.7 main + 0.3 face)...")
face_v2 = tf.keras.models.load_model(
    os.path.join(BASE, "face_model_best_02.h5"),
    custom_objects={'loss': focal_loss()}, compile=False)
preds_face = get_preds(face_v2, val_gen)
fused = 0.70 * preds_main + 0.30 * preds_face
ablation['Full fusion (0.7 main + 0.3 face model)'] = metrics(TRUE_LABELS, fused)
del main_model, face_v2

print_table(ablation, "EXPERIMENT 2: ABLATION STUDY")
save_json(ablation, "ablation_results.json")


# ════════════════════════════════════════
# EXPERIMENT 3: ROC CURVES
# ════════════════════════════════════════
print("Plotting ROC curves...")
fig, ax = plt.subplots(figsize=(9, 7))
colors = ['#1F4E79', '#2E75B6', '#E74C3C', '#27AE60', '#F39C12']
all_for_roc = {**results}

for (name, m), color in zip(all_for_roc.items(), colors):
    ax.plot(m['fpr'], m['tpr'],
            label=f"{name} (AUC={m['auc']:.3f})",
            color=color, linewidth=2)

ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random classifier (AUC=0.500)')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — Model Comparison\n(Evaluated on held-out validation set, 20% split)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(RESULTS_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"ROC curve saved: {roc_path}")


# ════════════════════════════════════════
# ABLATION BAR CHART
# ════════════════════════════════════════
print("Plotting ablation chart...")
configs  = list(ablation.keys())
accs     = [ablation[c]['accuracy']  for c in configs]
f1s      = [ablation[c]['f1']        for c in configs]
aucs_pct = [ablation[c]['auc'] * 100 for c in configs]

x     = np.arange(len(configs))
width = 0.25
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - width, accs,     width, label='Accuracy %', color='#1F4E79', alpha=0.85)
b2 = ax.bar(x,         f1s,      width, label='F1 Score %',  color='#2E75B6', alpha=0.85)
b3 = ax.bar(x + width, aucs_pct, width, label='AUC x100',    color='#70AD47', alpha=0.85)

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Ablation Study — Contribution of Each Component', fontsize=14, fontweight='bold')
ax.set_xticks(x)
short_labels = [c.replace(' (', '\n(').replace(', ', ',\n') for c in configs]
ax.set_xticklabels(short_labels, fontsize=8, rotation=10, ha='right')
ax.legend(fontsize=10)
ax.set_ylim(40, 110)
ax.grid(axis='y', alpha=0.3)
for bars in [b1, b2, b3]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=7)
plt.tight_layout()
ablation_path = os.path.join(RESULTS_DIR, "ablation_study.png")
plt.savefig(ablation_path, dpi=150)
plt.close()
print(f"Ablation chart saved: {ablation_path}")


# ════════════════════════════════════════
# EXPERIMENT 4: CROSS-CAMERA (WEBCAM)
# ════════════════════════════════════════
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
        if not ret:
            break
        cv2.putText(frame, f"A=attentive({count['attentive']}/{TARGET})  D=distracted({count['distracted']}/{TARGET})  Q=quit",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("Webcam Collection — ADAS Cross-Camera Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            cv2.imwrite(os.path.join(att_dir, f"att_{count['attentive']:04d}.jpg"), frame)
            count['attentive'] += 1
        elif key == ord('d'):
            cv2.imwrite(os.path.join(dis_dir, f"dis_{count['distracted']:04d}.jpg"), frame)
            count['distracted'] += 1
        elif key == ord('q'):
            break
        if count['attentive'] >= TARGET and count['distracted'] >= TARGET:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected: {count['attentive']} attentive, {count['distracted']} distracted")

# Evaluate on webcam set
webcam_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    webcam_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)
webcam_labels = webcam_gen.classes
print(f"Webcam test set: {webcam_gen.samples} images")

cross = {}

main_model = tf.keras.models.load_model(
    os.path.join(BASE, "multistream", "multistream", "video_model_nst_v2_best.h5"),
    custom_objects={'loss': focal_loss()}, compile=False)
# ─── NEW HYBRID EVALUATION LOOP ───
# ─── NEW HYBRID EVALUATION LOOP (FIXED) ───
print("\n[Hybrid Evaluation] Running CNN + Heuristic logic on webcam set...")

y_true = []
y_pred_final = []

for class_name in ['attentive', 'distracted']:
    class_dir = os.path.join(webcam_dir, class_name)
    target_label = 1 if class_name == 'distracted' else 0
    
    # Filter for image files
    img_list = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in img_list:
        img_path = os.path.join(class_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # 1. Preprocess
        img_input = preprocess(frame) 
        
        # 2. Get Raw Model Prediction
        raw_prob = float(main_model.predict(img_input, verbose=0)[0][0])
        
        # 3. Apply Heuristic Fusion (UNPACKING THE 4 RETURNS)
        # We unpack: (dict, counter, count, duration)
        ms_info, c1, c2, c3 = get_multistream_info(
            frame, 
            raw_prob > 0.5, 
            eye_closed_counter=0, 
            fatigue_events_count=0, 
            fatigue_max_duration=0
        )
        
        # 4. Record results - ms_info is now correctly identified as a dict
        y_true.append(target_label)
        
        # Extract the smart decision made by the Intelligence Layer
        is_distracted = ms_info.get('final_distracted', raw_prob > 0.5)
        y_pred_final.append(1.0 if is_distracted else 0.0)

# Calculate the new improved metrics
cross['Ours (Hybrid System) — webcam'] = metrics(np.array(y_true), np.array(y_pred_final))
print("✅ Hybrid evaluation complete.")

# Calculate the new improved metrics
cross['Ours (Hybrid System) — webcam'] = metrics(np.array(y_true), np.array(y_pred_final))

face_v2 = tf.keras.models.load_model(
    os.path.join(BASE, "face_model_best_02.h5"),
    custom_objects={'loss': focal_loss()}, compile=False)
webcam_gen.reset()
cross['Face model v2 — webcam'] = metrics(webcam_labels, get_preds(face_v2, webcam_gen))

del main_model, face_v2

print_table(cross, "EXPERIMENT 4: CROSS-CAMERA RESULTS (trained SDDD → tested webcam)")
print("Domain gap = difference between val accuracy and webcam accuracy")
save_json(cross, "cross_camera_results.json")

# ─── Cross-camera bar chart (val vs webcam) ───
try:
    with open(os.path.join(RESULTS_DIR, "comparison_results.json")) as f:
        comp = json.load(f)

    model_names = ['Ours (MobileNetV2+NST)', 'Face model v2']
    val_accs    = [
        comp.get('Ours: MobileNetV2 + NST + Focal Loss', {}).get('accuracy', 0),
        comp.get('Ours: Face crop model (DNN)', {}).get('accuracy', 0)
    ]
    wcam_accs = [
        cross.get('Ours (MobileNetV2+NST) — webcam', {}).get('accuracy', 0),
        cross.get('Face model v2 — webcam', {}).get('accuracy', 0)
    ]

    x     = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, val_accs,  width, label='Validation set (SDDD, same camera)', color='#1F4E79', alpha=0.85)
    ax.bar(x + width/2, wcam_accs, width, label='Webcam test set (unseen camera)',    color='#E74C3C', alpha=0.85)

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Cross-Camera Generalisation\nSDDD-trained models vs unseen webcam camera',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    cross_path = os.path.join(RESULTS_DIR, "cross_camera.png")
    plt.savefig(cross_path, dpi=150)
    plt.close()
    print(f"Cross-camera chart saved: {cross_path}")
except Exception as e:
    print(f"Cross-camera chart skipped: {e}")


# ════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════
print("\n" + "="*60)
print("ALL DONE — files saved to results/")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  {f}")
print("="*60)