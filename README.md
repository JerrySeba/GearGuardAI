# ⚙️ GearGuard AI — Industrial Gearbox Multi-Fault Condition Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Dark-36e6f5.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GearGuard AI** is a software-only, edge-deployable rotating machinery health intelligence platform designed for real-time gearbox fault diagnosis under variable speed and dynamic torque loads with limited failure data.

---

## 🌟 Key Highlights & Innovations

1. **7-Class Industrial Fault Taxonomy:**
   - Isolates fine-grained failure modes beyond binary health detection:
     - `Healthy Baseline`
     - `Broken Tooth`
     - `Chipped Tooth`
     - `Missing Tooth`
     - `Tooth Root Crack`
     - `Surface Wear / Pitting`
     - `Bearing Fault`

2. **Universal Multi-Channel Adaptation ($C \ge 1$):**
   - Seamlessly ingests single-channel ($C=1$), dual-shaft ($C=2$), triaxial ($C=3$), SpectraQuest ($C=4$), and SEU DDS ($C=8$) vibration arrays.
   - Dynamic channel projection maps any transducer configuration into an invariant 4-virtual-channel representation.

3. **Multi-Class Fault Probability Spectrum:**
   - Computes calibrated probability distributions across all 7 classes per segment and file.
   - Highlights primary diagnosis and secondary candidate failure modes with exact likelihoods.

4. **Multi-Format Ingestion & Micro-Telemetry Support:**
   - Reads `.csv`, `.tsv`, `.txt`, `.dat`, and MATLAB `.mat` files (classical v5/v7 and modern v7.3 HDF5).
   - Automatically detects and transposes horizontal matrices.
   - Strips 0–5V TTL tachometer square waves ($K < 1.35, \text{PTP} > 3.0\text{V}$).
   - Employs **Harmonic Spline Expansion** to evaluate micro-telemetry recordings down to 10 samples with zero crashes.

5. **Physics-Grounded Diagnostic Reasoning Engine:**
   - Dynamically translates numerical metrics (Kurtosis, Crest Factor, RMS, Peak $g$) into actionable engineering explanations:
     - Primary cause of failure
     - Key warning points
     - Telemetry evidence breakdown
     - Failure progression & degradation mechanics
     - Prescribed maintenance & inspection protocols

6. **Cyber-Industrial SCADA Dashboard:**
   - 100% offline edge deployment (zero cloud / Wi-Fi dependency).
   - High-speed Plotly time-domain waveforms, FFT power spectra, and multi-channel overlays.
   - Dedicated `Sensors & Accelerometers` hardware spec page.
   - Segment-by-segment diagnosis breakdown with CSV audit export.

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
├── app.py                                   # Master Streamlit SCADA application & adaptive engine
├── gearguard_real_model.joblib              # Calibrated 500-tree voting ensemble (72 MB compressed)
├── train_universal_model.py                 # Multi-domain scale-invariant training pipeline (9,830 vectors)
├── train_real_model.py                      # Original SpectraQuest training script
├── test_variable_load.py                    # Cross-load generalization evaluation script
├── requirements.txt                         # Python dependencies
├── README.md                                # Repository overview and quickstart guide
├── PROJECT_IMPLEMENTATION_AND_ARCHITECTURE_PLAN.md  # Comprehensive architectural master report
└── samples/                                 # Universal benchmark test sample suite
    ├── normal_0.mat                         # CWRU healthy baseline (.mat)
    ├── B007_0.mat                           # CWRU ball fault (.mat)
    ├── IR007_0.mat                          # CWRU inner race fault (.mat)
    ├── OR007@6_0.mat                        # CWRU outer race fault (.mat)
    ├── SEU_Health_20_0.csv                  # SEU healthy baseline (8-ch CSV)
    ├── SEU_Chipped_20_0.csv                 # SEU chipped tooth (8-ch CSV)
    ├── SEU_Miss_20_0.csv                    # SEU missing tooth (8-ch CSV)
    ├── SEU_Root_20_0.csv                    # SEU root crack (8-ch CSV)
    ├── SEU_Surface_20_0.csv                 # SEU surface wear (8-ch CSV)
    ├── healthy_4ch_sample.txt               # SpectraQuest healthy sample
    ├── broken_4ch_sample.txt                # SpectraQuest broken tooth sample
    ├── chipped_1ch_sample.csv               # Single-channel chipped tooth sample
    ├── micro_10sample_healthy.txt           # 10-point micro-sample healthy baseline
    └── micro_10sample_broken.txt            # 10-point micro-sample broken tooth
```

---

## 📊 Problem Statement Alignment

| Criterion | GearGuard AI Implementation |
| :--- | :--- |
| **Fault Isolation** | 7 Discrete Mechanical Classes (Healthy, Broken, Chipped, Missing, Crack, Wear, Bearing) |
| **Variable Speed & Load** | Tested across 0% to 90% dynamic torque loads and fluctuating shaft speeds (20–50 Hz) |
| **Multi-Channel Adaptation** | Universal channel projector dynamically handles $1, 2, 3, 4, 8+$ channels |
| **Confidence & Probability** | Calibrated probability spectrum per segment and whole file |
| **Micro-Telemetry Support** | Physics-guided harmonic spline expansion handles short bursts down to 10 points |
| **Hardware Requirement** | Software-only; runs locally on standard laptop CPU in $< 20\text{ ms}$ |

---

*GearGuard AI — Software-Only Industrial Machinery Health Intelligence.*
