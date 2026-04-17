# ADAS — Advanced Driver Assistance System
### Driver Distraction Detection using Deep Learning & Explainable AI

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-18%20(Vite)-61dafb)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black)](https://flask.palletsprojects.com)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-black)](https://adas-advanced-driver-assistance-sys.vercel.app)

---

## Overview

This project presents a **production-grade, multi-stream driver distraction detection system** built on MobileNetV2 transfer learning with a novel intelligence layer for real-time analysis. It detects driver distraction from both uploaded videos and live webcam feeds, providing explainability through Grad-CAM visualizations, sub-model signals, fatigue tracking, and a safety grading system.

The system is motivated by road safety concerns — driver distraction is one of the leading causes of road accidents worldwide. Beyond basic classification, this project explicitly addresses the **dashcam-to-webcam domain gap** through Neural Style Transfer augmentation, a rule-based heuristic fusion engine, temporal smoothing, and a fatigue accumulator.

> 🌐 **Live Demo:** [adas-advanced-driver-assistance-sys.vercel.app](https://adas-advanced-driver-assistance-sys.vercel.app)

---

## Microservices Architecture

The system runs as **three independent services**, each in its own Python environment to avoid dependency conflicts between TensorFlow, YOLOv8, and MediaPipe.

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│                   (Vite — localhost:5173)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP REST
┌────────────────────────────▼────────────────────────────────────┐
│              app.py — Main Flask Backend                        │
│              backend_env  |  localhost:5000                     │
│                                                                 │
│  • MobileNetV2 primary classifier (video + live)                │
│  • Multistream sub-models (head / eye / mouth)                  │
│  • Grad-CAM explainability                                      │
│  • Temporal smoothing + fatigue accumulator                     │
│  • Safety grade engine                                          │
│  • Session management + PDF report generation                   │
│  • Recording / trimming pipeline                                │
│                                                                 │
│       calls ↓ HTTP (0.8s timeout)    calls ↓ HTTP (0.3s timeout)│
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────────────┐    │
│  │    yolo_server.py    │  │    mediapipe_server.py       │    │
│  │    yolo_env          │  │    mediapipe_env             │    │
│  │    localhost:5001    │  │    localhost:5002            │    │
│  │                      │  │                              │    │
│  │  YOLOv8 object       │  │  MediaPipe Face Mesh         │    │
│  │  detection:          │  │  landmark analysis:          │    │
│  │  • Phone detected    │  │  • EAR (Eye Aspect Ratio)    │    │
│  │  • Drinking detected │  │  • MAR (Mouth Aspect Ratio)  │    │
│  └──────────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Why Three Environments?

| Service | Environment | Reason for Isolation |
|---------|------------|----------------------|
| `app.py` | `backend_env` | TensorFlow 2.13 + NumPy 1.24.3 (pinned) conflicts with YOLO & MediaPipe |
| `yolo_server.py` | `yolo_env` | Ultralytics YOLOv8 requires its own dependency tree |
| `mediapipe_server.py` | `mediapipe_env` | MediaPipe has strict NumPy/OpenCV version requirements |

Both microservices are **non-blocking** — if either times out, `app.py` gracefully falls back to sub-model and primary model predictions without crashing the pipeline.

---

## Full System Pipeline

```
Input (Video Upload / Webcam Frame)
              ↓
  OpenCV DNN Face Detector (Caffe — res10_300x300_ssd_iter_140000)
  Fallback: Haar Cascade
              ↓
  ┌──────────────────────────────────────────────────────────┐
  │                  app.py inference                        │
  │                                                          │
  │  ┌─────────────────┐  ┌──────────┐  ┌───────┐  ┌──────┐│
  │  │  Primary Model  │  │   Head   │  │  Eye  │  │Mouth ││
  │  │  MobileNetV2    │  │  Pose    │  │ State │  │State ││
  │  │  10 classes     │  │ 9 classes│  │6 class│  │3 cls ││
  │  │  95.42% acc     │  │          │  │       │  │      ││
  │  └─────────────────┘  └──────────┘  └───────┘  └──────┘│
  └──────────────────────────────────────────────────────────┘
              ↓                         ↓
  ┌───────────────────┐     ┌───────────────────────┐
  │  yolo_server.py   │     │  mediapipe_server.py  │
  │  localhost:5001   │     │  localhost:5002        │
  │                   │     │                        │
  │  Phone detected?  │     │  EAR → blink/drowsy    │
  │  Drinking?        │     │  MAR → yawn/mouth open │
  └────────┬──────────┘     └──────────┬─────────────┘
           └──────────────┬────────────┘
                          ↓
  ┌────────────────────────────────────┐
  │          Intelligence Layer        │
  │                                    │
  │  Heuristic Fusion Engine           │
  │  Temporal Smoothing (15 frames)    │
  │  Fatigue Accumulator               │
  │  EAR / MAR threshold logic         │
  └─────────────────┬──────────────────┘
                    ↓
  Safety Grade (A–F) + Distraction Alert + Reasons
```

---

## Three-Phase Classification Pipeline

```
Phase 1 → Binary Classification      (Safe vs Distracted)          97%    State Farm dataset
Phase 2 → Multiclass Classification  (10 distraction types)        94%    State Farm dataset
Phase 3 → Video + NST Augmentation   (real-world generalization)   95.42% SDDD + NST
```

### Phase 1 — Binary Classification
- **Model:** MobileNetV2 + GlobalAveragePooling + Dense layers
- **Dataset:** State Farm Distracted Driver Detection (10 classes collapsed to 2)
- **Classes:** Safe (c0) vs Distracted (c1–c9)
- **Accuracy:** 97%
- **Loss:** Focal loss (γ=2, α=0.25)

### Phase 2 — Multiclass Classification
- **Model:** MobileNetV2 (fine-tuned)
- **Classes:** Safe, Texting (Right), Phone (Right), Texting (Left), Phone (Left), Radio, Drinking, Reaching Behind, Hair/Makeup, Talking to Passenger
- **Accuracy:** 94%

### Phase 3 — Video Detection with Neural Style Transfer
- **Model:** MobileNetV2 retrained on SDDD + NST-augmented data (`video_model_nst_v2_best.h5`)
- **Training:** Two-phase fine-tuning (freeze → unfreeze last 30 layers)
- **Accuracy:** 95.42% | AUC: 0.9947
- **Key upgrade:** NST augmentation boosted accuracy from 82% → 95.42%

---

## Key Innovations

### Neural Style Transfer (NST) Augmentation

**Problem:** Model overfitting to dashcam-specific backgrounds — ~46 percentage point accuracy drop on webcam input.

**Solution:** VGG19-based NST generates domain-diverse synthetic training images, forcing the model to focus on driver actions rather than car interior backgrounds.

```
VGG19 Feature Extraction
        ↓
Gram Matrix Style Representation
        ↓
Content Loss + Style Loss Optimization
        ↓
Diverse synthetic training samples
(Applied to training set only)
```

**Impact:** 82% → 95.42% validation accuracy, AUC 0.9947

---

### Intelligence Layer

#### Heuristic Fusion Engine
Combines outputs from all models with domain-aware priority logic:

```
YOLO phone detected                           → Distracted = True
YOLO drinking + mouth "Wide Open"             → Distracted = True
MAR > 0.6 (MediaPipe yawn)                   → Distracted = True
EAR < 0.25 for 3+ consecutive frames          → Fatigue = True → Distracted
Head non-Frontal (Left/Right/Down etc.)       → Distracted = True
Head Frontal + Eye Front/Up + no alerts       → Distracted = False
Otherwise                                     → defer to primary model
```

Additional heuristics:
- **Phone Usage Suspected** — head Left/Right + eyes Left/Right/Front for 5+ frames
- **Drinking Suspected (Pose)** — head Up/Diagonal + mouth "Wide Open" + MAR > 0.40
- **Eating Suspected** — mouth "Wide Open" + head Frontal/Down + 0.40 < MAR ≤ 0.60

#### Temporal Smoothing Buffer
Rolling 15-frame window to suppress per-frame noise:
```python
buffer = deque(maxlen=15)
# Distracted if ≥ 12/15 frames distracted
# OR ≥ 9/15 total AND ≥ 3/5 recent frames distracted
# Clear attention override if head Frontal + eye Front/Up + 0 recent distracted
```

#### Fatigue Accumulator
```python
FATIGUE_THRESHOLD = 3  # ~3 consecutive seconds at 1 FPS live capture
if EAR < 0.25 or eye_model == "Closed":
    eye_closed_counter += 1
if eye_closed_counter == FATIGUE_THRESHOLD:
    fatigue_events_count += 1   # trigger "Fatigue Detected"
fatigue_max_duration = max(fatigue_max_duration, eye_closed_counter)
# Yawning (MAR > 0.6) is tracked separately — does not increment fatigue counter
```

#### Safety Grade System
| Grade | Score | Meaning |
|-------|-------|---------|
| A | 90–100 | Excellent — Focused |
| B | 75–89 | Good — Minor Gaps |
| C | 60–74 | Caution — Frequent Distraction |
| F | < 60 | DANGEROUS — Immediate Risk |

`score = max(0, 100 − distracted_pct − (fatigue_events × 15))`

---

### Ensemble Prediction (Live Mode)

Live webcam uses a weighted ensemble for better webcam-domain generalization:

```
prob = 0.60 × SDDD_model + 0.40 × custom_model
Distraction threshold (live): 0.65
Distraction threshold (batch video): 0.50
```

`custom_model` (`video_model_custom_final.h5`) is trained on webcam-domain data. Falls back to SDDD model only if unavailable.

---

### YOLO Service — `yolo_server.py`

**Environment:** `yolo_env` | **Port:** `5001`

Runs YOLOv8 for direct object detection. Called by `app.py` via HTTP POST with a 0.8s timeout — silently returns `false` on failure so the live stream is never blocked.

**Detects:** mobile phone in hand, bottle/cup near face (drinking)

**Endpoint:** `POST /detect`
```json
// Request
{ "image": "<base64_jpg>" }
// Response
{ "phone": true, "drinking": false }
```

---

### MediaPipe Service — `mediapipe_server.py`

**Environment:** `mediapipe_env` | **Port:** `5002`

Runs MediaPipe Face Mesh for precise facial landmark measurements. Called by `app.py` via HTTP POST with a 0.3s timeout.

**Computes:**
- **EAR (Eye Aspect Ratio)** — blink/eye closure detection (threshold: `< 0.25`)
- **MAR (Mouth Aspect Ratio)** — yawn detection (threshold: `> 0.6`)

**Endpoint:** `POST /analyze`
```json
// Request
{ "image": "<base64_jpg>" }
// Response
{ "ear": 0.19, "mar": 0.68 }
```

EAR/MAR values are also returned to the frontend per frame for full transparency.

---

## Domain Gap Analysis

| Experiment | Setup | Accuracy |
|------------|-------|----------|
| Baseline In-Domain | Standard SDDD validation set | 95.42% |
| Zero-Shot Webcam | Webcam frames, no tuning | ~49% |
| Tuned Live Mode | Webcam + ensemble + thresholds + fusion | Improved |
| Ablation (no NST) | Remove NST, retrain from scratch | 82% |

**Root causes of gap:** camera angle, indoor lighting, home/office backgrounds, face size and resolution differences between dashcam and webcam setups.

---

## Results Summary

| Model | Accuracy | AUC | Dataset |
|-------|----------|-----|---------|
| Binary Classification | **97%** | — | State Farm |
| Multiclass Classification | **94%** | — | State Farm |
| Video Detection (with NST) | **95.42%** | **0.9947** | SDDD + NST |
| Eye Gaze Sub-model | **90%** | — | SDDD |
| Mouth State Sub-model | **75%** | — | SDDD |
| Head Pose Sub-model | **67%** | — | SDDD |
| Face Binary Classifier | **76.4%** | — | SDDD |

---

## Project Structure

```
ADAS_PROJECT/
│
├── 📄 app.py                          # Main Flask backend (port 5000, backend_env)
├── 📄 yolo_server.py                  # YOLO microservice (port 5001, yolo_env)
├── 📄 mediapipe_server.py             # MediaPipe microservice (port 5002, mediapipe_env)
│
├── 📄 model.py                        # Phase 1: Binary model training
├── 📄 train_multiclass_model.py       # Phase 2: Multiclass training
├── 📄 train_video_model.py            # Phase 3: Video model training
├── 📄 neural_style_transfer.py        # NST augmentation (VGG19)
├── 📄 prepare_video_dataset.py        # Frame extraction (5 FPS sampling)
├── 📄 fix_attentive.py                # Dataset balancing
├── 📄 video_detection.py              # Standalone real-time video detection
├── 📄 preprocessing.py                # Face detection + crop pipeline
├── 📄 evaluate_binary_model.py        # Binary model evaluation
├── 📄 evaluate_multiclass_model.py    # Multiclass evaluation
├── 📄 generate_video_results.py       # Results generation
├── 📄 explainable_distraction.py      # Grad-CAM explainability
│
├── 📄 video_model_custom_final.h5     # Webcam-domain model (live ensemble)
│
├── 📁 multistream/multistream/        # Trained model weights
│   ├── video_model_nst_v2_best.h5    # Primary video model (NST v2)
│   ├── head_model_best.h5            # Head pose model (9 classes)
│   ├── eye_model_best.h5             # Eye gaze model (6 classes)
│   └── mouth_model_best.h5           # Mouth state model (3 classes)
│
├── 📁 multistream/                    # Intelligence layer + sub-model training
│   ├── train_head_model.py
│   ├── train_eye_model.py
│   ├── train_mouth_model.py
│   ├── fusion_combined.py             # Heuristic fusion + safety grade
│   ├── temporal_smoothing.py          # Rolling prediction buffer
│   ├── fatigue_accumulator.py         # Long-term fatigue tracking
│   └── video_detection_multistream.py
│
├── 📁 adas-dashboard/                 # React + Vite frontend
│   └── src/App.js                     # Image / Video / Live tabs
│
├── 📁 results/                        # Phase 1 & 2 results
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   └── loss_curve.png
│
└── 📁 video_results/                  # Phase 3 results
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── classification_report.txt
    └── sample_predictions.png
```

---

## Getting Started

### Prerequisites

```
Python 3.11
Node.js 18+
Three separate Python virtual environments (see below)
```

### Environment Setup

```bash
# 1. Main backend environment
python -m venv backend_env
backend_env\Scripts\activate          # Windows
source backend_env/bin/activate        # Mac/Linux
pip install -r requirements_backend.txt

# 2. YOLO environment
python -m venv yolo_env
yolo_env\Scripts\activate
pip install -r requirements_yolo.txt

# 3. MediaPipe environment
python -m venv mediapipe_env
mediapipe_env\Scripts\activate
pip install -r requirements_mediapipe.txt
```

### Clone the Repo

```bash
git clone https://github.com/Esther-Graceia-Precious/ADAS-Advanced-driver-assistance-system.git
cd ADAS-Advanced-driver-assistance-system
```

### Running the System

All three services must be running simultaneously. Open four terminals:

**Terminal 1 — MediaPipe Server** *(start first)*
```bash
mediapipe_env\Scripts\activate
python mediapipe_server.py
# Listening on http://127.0.0.1:5002
```

**Terminal 2 — YOLO Server**
```bash
yolo_env\Scripts\activate
python yolo_server.py
# Listening on http://127.0.0.1:5001
```

**Terminal 3 — Main Backend**
```bash
backend_env\Scripts\activate
python app.py
# Listening on http://0.0.0.0:5000
```

**Terminal 4 — React Frontend**
```bash
cd adas-dashboard
npm install
npm run dev
# Open http://localhost:5173
```

> **Note:** The main backend will still run if YOLO or MediaPipe servers are unavailable — both calls have short timeouts and fail gracefully, falling back to sub-model predictions.

### Run Multistream Detection (Standalone)
```bash
backend_env\Scripts\activate
python multistream/video_detection_multistream.py
```

---

## Model Architecture

```
Input (224×224×3)
        ↓
OpenCV DNN Face Detector → Face Crop
(res10_300x300_ssd_iter_140000.caffemodel, conf > 0.5, pad = 20px)
Fallback: Haar Cascade frontal face
        ↓
MobileNetV2 (ImageNet pretrained)
        ↓
GlobalAveragePooling2D
        ↓
Dense(512, ReLU)
        ↓
Dropout(0.5)
        ↓
Dense(1, Sigmoid)      ← Binary
Dense(10, Softmax)     ← Multiclass / Video
```

**Training Strategy:**
- Phase 1: Freeze base model, train custom head (lr=1e-4, 10 epochs)
- Phase 2: Unfreeze top 30 layers, fine-tune (lr=1e-5, 10 epochs)
- Loss: Focal loss (γ=2, α=0.25)
- Augmentation: Rotation, flip, zoom + Neural Style Transfer (VGG19)

---

## API Reference

### Main Backend — `app.py` (port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze_image` | POST | Single image analysis with Grad-CAM + multistream |
| `/analyze` | POST | Full video file analysis with timeline + alerts |
| `/live_start` | POST | Initialize a live webcam session → returns `session_id` |
| `/analyze_live` | POST | Analyze one live webcam frame (base64) |
| `/live_end` | POST | End session, return full summary + save for PDF |
| `/recording_start` | POST | Begin buffering annotated frames to memory |
| `/recording_stop` | POST | Write buffer to MP4, return filename |
| `/download_video/<filename>` | GET | Download recorded MP4 |
| `/download_report/<session_id>` | GET | Download ReportLab PDF safety report |
| `/trim_video` | POST | Trim recorded video to specified time segments |

### YOLO Server — `yolo_server.py` (port 5001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | YOLOv8 phone/drinking detection |

### MediaPipe Server — `mediapipe_server.py` (port 5002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | EAR + MAR from Face Mesh landmarks |

### Live Frame Response
```json
{
  "label": "Distracted",
  "confidence": 78.4,
  "risk_level": "HIGH",
  "multistream": {
    "face_detected": true,
    "head": "Left",
    "eye": "Closed",
    "mouth": "Slight Open",
    "reasons": ["Looking Left", "Eyes Closed (EAR)", "Phone Detected"],
    "ear": 0.19,
    "mar": 0.22
  },
  "session": {
    "distracted_pct": 61.0,
    "fatigue_events": 2,
    "eye_closed_frames": 4,
    "safety_score": 9.0,
    "safety_grade": "F",
    "safety_message": "DANGEROUS — Immediate Risk"
  }
}
```

---

## PDF Report Generation

At the end of each live session, a professional PDF is generated via **ReportLab** and served from `/download_report/<session_id>`.

**Report includes:**
- Session summary (frames analyzed, attentive/distracted %, risk level)
- Safety assessment (grade A–F, score out of 100, message)
- Fatigue analysis (event count, max consecutive closed-eye duration)
- Grad-CAM visualization — original frame vs. attention heatmap for the highest-confidence distracted moment

---

## Explainability

- **Grad-CAM:** Applied to the most distracted frame in both video and live sessions — highlights what the model focused on
- **Multistream reasons:** Per-frame human-readable tags — "Looking Left", "Eyes Closed (EAR)", "Phone Detected", "Yawning Detected", "Drinking Suspected (Pose)"
- **EAR/MAR values:** Raw MediaPipe landmark metrics returned per frame
- **Safety Grade:** Quantified A–F attention score with message
- **Distraction Timeline:** Per-frame confidence graph + timestamped alert segments (video mode)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow 2.13, Keras |
| Primary Model | MobileNetV2 (Transfer Learning) |
| Object Detection | YOLOv8 (Ultralytics) — `yolo_env` |
| Face Landmarks | MediaPipe Face Mesh — `mediapipe_env` |
| Augmentation | Neural Style Transfer (VGG19) |
| Face Detection | OpenCV DNN (Caffe res10 model) + Haar fallback |
| Backend | Flask 3.0, OpenCV — `backend_env` |
| Frontend | React 18, Vite, Recharts, Axios |
| PDF Reports | ReportLab |
| Explainability | Grad-CAM |
| Deployment | Vercel (frontend) |
| Datasets | State Farm Distracted Driver, SDDD (SafeDriveMetrics) |

---

## Future Work

- Docker Compose setup for single-command startup of all three services
- Webcam-domain training data collection for full domain adaptation
- TensorFlow Lite quantization for edge deployment (target: < 50ms/frame)
- WebSocket live stream for lower latency than REST polling
- Attention mechanism replacing MobileNetV2 backbone
- Occlusion robustness (masks, sunglasses, low-light conditions)
- Cloud deployment after `BASE_DIR` path migration

---

## Author

**Esther Graceia Precious**
ADAS — Advanced Driver Assistance System
Driver Distraction Detection using Deep Learning & Explainable AI
