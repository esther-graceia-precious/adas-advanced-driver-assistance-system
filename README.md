# ADAS — Advanced Driver Assistance System
### Driver Distraction Detection using Deep Learning & Explainable AI

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-18-61dafb)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black)](https://flask.palletsprojects.com)

---

## Overview

This project presents a **three-phase driver distraction detection system** built using MobileNetV2 transfer learning. It detects whether a driver is distracted in real time using video input, and provides detailed explainability through Grad-CAM visualizations and a multistream analysis pipeline.

The system is motivated by road safety concerns — driver distraction is one of the leading causes of road accidents worldwide. Our approach combines classical image classification with video-based detection and a novel multistream architecture that analyzes head pose, eye gaze, and mouth state simultaneously.

---

## Three-Phase Architecture

```
Phase 1 → Binary Classification       (Safe vs Distracted)         97% accuracy
Phase 2 → Multiclass Classification   (10 distraction types)       94% accuracy
Phase 3 → Video Detection             (real-time video analysis)   82% accuracy
```

### Phase 1 — Binary Classification
- **Model:** MobileNetV2 + GlobalAveragePooling + Dense layers
- **Dataset:** Standard driver distraction image dataset (10 classes collapsed to 2)
- **Classes:** Safe (c0) vs Distracted (c1–c9)
- **Accuracy:** 97%
- **Loss:** Focal loss to handle class imbalance

### Phase 2 — Multiclass Classification
- **Model:** MobileNetV2 (fine-tuned)
- **Classes:** Safe, Texting (Right), Phone (Right), Texting (Left), Phone (Left), Radio, Drinking, Reaching Behind, Hair/Makeup, Talking to Passenger
- **Accuracy:** 94%

### Phase 3 — Video Detection
- **Model:** MobileNetV2 retrained on SDDD (SafeDriveMetrics) video dataset
- **Dataset:** SDDD — recorded in a simulated driving environment
- **Training:** Two-phase fine-tuning (freeze → unfreeze last 30 layers)
- **Accuracy:** 82% on real driving videos
- **Features:** Real-time alert, confidence timeline, Grad-CAM explainability

---

## Multistream Architecture (Novel Contribution)

To address the **domain gap problem** between static images and real driving videos, we developed a multistream pipeline that analyzes facial features separately:

```
Video Frame
     ↓
Face Detection (OpenCV Haar Cascade)
     ↓
┌──────────┐  ┌──────────┐  ┌──────────┐
│   Head   │  │   Eye    │  │  Mouth   │
│  Pose    │  │  Gaze    │  │  State   │
│ 9 classes│  │ 6 classes│  │ 3 classes│
│   67%    │  │   90%    │  │   75%    │
└──────────┘  └──────────┘  └──────────┘
     ↓
Fusion Logic → Distraction Reason
"Phone use suspected / Head Left / Eyes Closed (Fatigue)"
```

**Multistream Model Accuracies:**
| Model | Classes | Accuracy |
|-------|---------|----------|
| Head Pose | 9 | 67% |
| Eye Gaze | 6 | 90% |
| Mouth State | 3 | 75% |

---

## Key Features

- **Three-phase detection** — binary, multiclass, and video
- **Real-time video detection** — frame-by-frame analysis with alerts
- **Grad-CAM explainability** — visual heatmaps showing model focus
- **Multistream analysis** — head, eye, and mouth detection
- **Risk level assessment** — LOW / MEDIUM / HIGH
- **Web dashboard** — upload image or video for analysis
- **Alert system** — audio alert when distraction detected
- **Distraction timeline** — confidence graph over video duration
- **Distraction events** — timestamped alert segments

---

## Results Summary

| Model | Accuracy | Dataset |
|-------|----------|---------|
| Binary Classification | **97%** | Standard image dataset |
| Multiclass Classification | **94%** | Standard image dataset |
| Video Detection | **82%** | SDDD video dataset |
| Eye Gaze | **90%** | SDDD image dataset |
| Mouth State | **75%** | SDDD image dataset |
| Head Pose | **67%** | SDDD image dataset |

---

## Project Structure

```
ADAS_PROJECT/
│
├── 📄 model.py                    # Phase 1: Binary model training
├── 📄 train_multiclass_model.py   # Phase 2: Multiclass training
├── 📄 train_video_model.py        # Phase 3: Video model training
├── 📄 prepare_video_dataset.py    # Frame extraction from videos
├── 📄 fix_attentive.py            # Dataset balancing
├── 📄 video_detection.py          # Real-time video detection
├── 📄 app.py                      # Flask backend API
├── 📄 preprocessing.py            # Data loading utilities
├── 📄 evaluate_binary_model.py    # Binary model evaluation
├── 📄 evaluate_multiclass_model.py# Multiclass evaluation
├── 📄 generate_video_results.py   # Results generation
├── 📄 explainable_distraction.py  # SHAP/LIME explainability
│
├── 📁 multistream/                # Novel multistream architecture
│   ├── train_head_model.py        # Head pose model (9 classes)
│   ├── train_eye_model.py         # Eye gaze model (6 classes)
│   ├── train_mouth_model.py       # Mouth state model (3 classes)
│   ├── fusion_combined.py         # Fusion logic with face detection
│   └── video_detection_multistream.py  # Multistream video detection
│
├── 📁 adas-dashboard/             # React frontend
│   └── src/App.js                 # Main dashboard (Image/Video/Live tabs)
│
├── 📁 results/                    # Phase 1 & 2 results
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   └── loss_curve.png
│
└── 📁 video_results/              # Phase 3 results
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── classification_report.txt
    └── sample_predictions.png
```

---

## Getting Started

### Prerequisites
```bash
Python 3.11
TensorFlow 2.13
Node.js 18+
```

### Installation

```bash
# Clone the repo
git clone https://github.com/Esther-Graceia-Precious/ADAS-Advanced-driver-assistance-system.git
cd ADAS-Advanced-driver-assistance-system

# Create virtual environment
python -m venv adas_env
adas_env\Scripts\activate   # Windows
source adas_env/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run Video Detection (Real-time)
```bash
python video_detection.py
```

### Run Flask Backend
```bash
python app.py
```

### Run React Dashboard
```bash
cd adas-dashboard
npm install
npm start
```
Open [http://localhost:3000](http://localhost:3000)

### Run Multistream Detection
```bash
python multistream/video_detection_multistream.py
```

---

## Model Architecture

```
Input (224×224×3)
        ↓
MobileNetV2 (ImageNet pretrained)
        ↓
GlobalAveragePooling2D
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.4)
        ↓
Dense(1, Sigmoid)     ← Binary
Dense(N, Softmax)     ← Multiclass
```

**Training Strategy:**
- Phase 1: Freeze base model, train top layers (lr=1e-4, 10 epochs)
- Phase 2: Unfreeze last 30 layers, fine-tune (lr=1e-5, 10 epochs)
- Loss: Focal loss (γ=2, α=0.25)
- Class weights for imbalanced datasets

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze_image` | POST | Analyze single image |
| `/analyze` | POST | Analyze video file |

### Image Analysis Response
```json
{
  "label": "Distracted",
  "confidence": 87.3,
  "risk_level": "HIGH",
  "gradcam": { "original": "...", "gradcam": "..." }
}
```

### Video Analysis Response
```json
{
  "summary": {
    "overall_status": "DISTRACTED",
    "attentive_pct": 42.0,
    "distracted_pct": 58.0,
    "alert_count": 3,
    "risk_level": "HIGH",
    "duration_seconds": 36.2
  },
  "frame_data": { "timestamps": [], "confidences": [], "labels": [] },
  "alerts": [{ "start": 12.5, "end": 18.3 }],
  "gradcam": { "original": "...", "gradcam": "..." }
}

---

## Known Limitations

1. **Domain gap:** Models trained on static images perform differently on video frames — addressed by training a dedicated video model on SDDD
2. **Background bias:** Grad-CAM occasionally highlights background instead of driver face — future work involves face cropping as preprocessing
3. **Phone detection:** Indirect detection via head/eye patterns rather than direct object detection

---

## Future Work

- Face cropping preprocessing to improve Grad-CAM focus
- Hand detection module for direct phone/object detection
- Attention mechanism replacing MobileNetV2
- Real-time webcam monitoring in dashboard
- PDF report generation for fleet managers

---

## Related Publication

> *"Utilizing Explainable AI to Decipher Transcriptomic Alterations in Pancreatic Cancer"*
> Published in **Elsevier Human Gene Journal, 2025**
> Demonstrates the application of XAI techniques for medical classification — methodologies extended to driver distraction detection in this project.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow 2.13, Keras |
| Model | MobileNetV2 (Transfer Learning) |
| Backend | Flask, OpenCV |
| Frontend | React, Recharts, Axios |
| Explainability | Grad-CAM |
| Dataset | SDDD (SafeDriveMetrics), Standard Driver Dataset |

---

## Author

**Esther Graceia Precious**
ADAS — Driver Distraction Detection System
