# ⚙️ GearGuard AI — Industrial Gearbox Condition Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Dark-36e6f5.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GearGuard AI** is a software-only, edge-deployable gearbox fault diagnosis platform designed for industrial condition monitoring (wind turbines, aerospace drivetrains, and manufacturing testbeds).

---

## 🌟 Key Highlights & Innovations

1. **Universal Multi-Channel Adaptation:**
   - Adapts seamlessly to **ANY number of sensor channels** ($C \ge 1$):
     - Single-channel vibration streams ($C=1$).
     - Dual-shaft accelerometer pairs ($C=2$).
     - NASA PHM 2009 multi-shaft recordings ($C=3$).
     - SpectraQuest testbed recordings ($C=4$).
     - Southeast University (SEU) DDS planetary rig ($C=8$).
   - Dynamic channel projection maps arbitrary inputs into statistical feature representations without throwing dimensional mismatch errors.

2. **Hybrid Diagnostic Engine:**
   - **Trained Real-Data Baseline:** 200-estimator Balanced Random Forest trained on physical benchmark data across variable operating torque loads ($0\%$ to $90\%$).
   - **Kinematic Physics Guard:** Fuses statistical descriptors (Pearson kurtosis, crest factor, RMS energy) with FFT harmonic spectral analysis to prevent false negatives on impact-heavy tooth shear faults.

3. **Cyber-Industrial SCADA Dashboard:**
   - Built with an industrial sci-fi dark theme (Orbitron typography, Plotly high-speed GPU rendering).
   - Real-time time-domain waveforms and FFT spectral analysis.
   - Dynamic multi-channel sensor subplots.
   - Segment-by-segment diagnosis breakdown with CSV audit export.
   - 100% offline edge deployment (zero cloud / Wi-Fi dependency).

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install the lightweight requirements:
```bash
git clone https://github.com/JerrySeba/GearGuardAI.git
cd GearGuardAI
pip install -r requirements.txt
```

### 2. Launch the Edge Dashboard
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 📂 Repository Structure

```
GearGuardAI/
├── app.py                      # Main Streamlit SCADA application & adaptive engine
├── gearguard_real_model.joblib # Trained machine learning model weights (Random Forest)
├── train_real_model.py         # Model training script with GroupShuffleSplit
├── test_variable_load.py       # Cross-load generalization evaluation script
├── samples/                    # Benchmark test recordings (1-channel & 4-channel)
│   ├── broken_4ch_sample.txt   # SpectraQuest 4-channel Broken Tooth recording
│   ├── healthy_4ch_sample.txt  # SpectraQuest 4-channel Healthy recording
│   └── chipped_1ch_sample.csv  # Single-channel incipient fault recording
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📊 Problem Statement Alignment

| Criterion | GearGuard AI Implementation |
| :--- | :--- |
| **Fault Isolation** | Healthy Baseline vs. Broken / Missing Gear Tooth |
| **Variable Speed & Load** | Tested and verified across $0\%$ to $90\%$ dynamic torque loading |
| **Multi-Channel Adaptation** | Universal channel projector dynamically handles $1, 2, 3, 4, 8+$ channels |
| **Confidence Scoring** | Segment-level model probabilities and overall mean diagnostic certainty |
| **Hardware Requirement** | Software-only; runs locally on standard laptop hardware in $< 15\text{ ms}$ |

---

*GearGuard AI — Software-Only Industrial Machinery Health Intelligence.*
