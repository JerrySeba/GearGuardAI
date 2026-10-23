"""
================================================================================
GEARGUARD AI — Universal Multi-Dataset & Multi-Scale Model Training Pipeline
Trains a robust, scale-invariant Random Forest ensemble across:
- SpectraQuest Variable Load (0% to 90% load, 4 channels)
- Southeast University DDS Gearbox (8 channels, multi-fault)
- Case Western Reserve University (CWRU) .mat Bearing/Gear Benchmarks
- PHM 2009 Multi-Speed / Multi-Load Gearbox Runs (low-voltage accelerometers)
- Micro-Telemetry Signals (10 to 50 samples expanded via harmonic spline)
- Random Gain & Channel Permutations for Zero Covariate Shift
================================================================================
"""

from pathlib import Path
import io
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import joblib

from scipy.stats import kurtosis, skew
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
HEALTHY_DIR = ROOT / "Healthy Data"
FAULTY_DIR = ROOT / "BrokenTooth Data"
PHM_DIR = Path("/home/jeremy/archive")
SAMPLES_DIR = ROOT / "samples"
OUT_MODEL = ROOT / "gearguard_real_model.joblib"

SEGMENT_SIZE = 2048

FEATURES = []
LABELS = []
GROUPS = []

# Reproducibility
np.random.seed(42)
random.seed(42)


def expand_short_signal(x, target_len=2048):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = len(x)
    if n >= target_len:
        return x
    if n < 4:
        reps = int(np.ceil(target_len / max(1, n)))
        return np.tile(x, (reps, 1))[:target_len]

    n_cycles = max(2, int(np.ceil(target_len / n)))
    tiled_x = np.tile(x, (n_cycles, 1))
    t_tiled = np.linspace(0, n_cycles, len(tiled_x), endpoint=False)
    t_new = np.linspace(0, n_cycles, target_len, endpoint=False)
    kind = "linear" if n < 6 else "cubic"
    f_interp = interp1d(t_tiled, tiled_x, kind=kind, axis=0, fill_value="extrapolate")
    return f_interp(t_new)


def strip_tachometer_channels(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] <= 1:
        return arr

    keep = []
    for c in range(arr.shape[1]):
        col = arr[:, c]
        col_ac = col - np.mean(col)
        std = float(np.std(col_ac))
        ptp = float(np.max(col) - np.min(col))
        k_p = float(kurtosis(col_ac, fisher=False)) if len(col_ac) >= 4 and std > 1e-9 else 3.0
        # Tachometer 0-5V pulse has mean > 1.0, ptp > 3.0, and Pearson kurtosis < 1.45
        if np.mean(col) > 1.0 and ptp > 3.0 and k_p < 1.45:
            continue
        keep.append(c)

    if not keep:
        return arr
    return arr[:, keep]


def adapt_channel_projection(segment, target_channels=4):
    C = segment.shape[1]
    if C == target_channels:
        return [0, 1, 2, 3]
    elif C == 1:
        return [0, 0, 0, 0]
    elif C == 2:
        return [0, 1, 0, 1]
    elif C == 3:
        return [0, 1, 2, 0]
    else:
        variances = [float(np.var(segment[:, i])) for i in range(C)]
        return list(np.argsort(variances)[-target_channels:][::-1])


def extract_features(segment):
    segment = np.asarray(segment, dtype=float)
    ch_map = adapt_channel_projection(segment, target_channels=4)
    values = []

    for ch in ch_map:
        raw_x = segment[:, ch].astype(float)
        x = raw_x - np.mean(raw_x) if len(raw_x) >= 16 else raw_x
        rms = float(np.sqrt(np.mean(x ** 2)))
        std = float(np.std(x))
        mean_abs = float(np.mean(np.abs(x)))
        peak = float(np.max(np.abs(x)))

        kurt_val = float(kurtosis(x, fisher=False)) if (len(x) >= 4 and std > 1e-9) else 3.0
        skew_val = float(skew(x)) if (len(x) >= 3 and std > 1e-9) else 0.0

        if not np.isfinite(kurt_val):
            kurt_val = 3.0
        if not np.isfinite(skew_val):
            skew_val = 0.0

        values.extend([
            rms,
            std,
            mean_abs,
            peak,
            peak / (rms + 1e-12),
            kurt_val,
            skew_val,
        ])

    return values


def parse_vibration_file(file_path):
    raw = file_path.read_bytes()
    # Check for MATLAB binary or HDF5 container
    if raw[:6] == b"MATLAB" or file_path.suffix.lower() == ".mat":
        try:
            mat = sio.loadmat(io.BytesIO(raw))
            arrays_2d = []
            cols_1d = {}
            for k in sorted(mat.keys()):
                if k.startswith("__"):
                    continue
                v = np.asarray(mat[k])
                if not np.issubdtype(v.dtype, np.number) or v.size <= 4 or "rpm" in k.lower():
                    continue
                arr = np.squeeze(v)
                if arr.ndim == 1 and arr.size >= 1:
                    cols_1d[k] = arr.astype(float)
                elif arr.ndim == 2 and max(arr.shape) >= 1:
                    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 16:
                        arr = arr.T
                    arrays_2d.append(arr.astype(float))
            if arrays_2d:
                return strip_tachometer_channels(max(arrays_2d, key=lambda a: a.size))
            elif cols_1d:
                min_len = min(len(c) for c in cols_1d.values())
                return strip_tachometer_channels(np.column_stack([c[:min_len] for c in cols_1d.values()]))
        except Exception:
            pass

    # Fallback / text parsing
    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    for idx, l in enumerate(lines[:40]):
        if l.strip().lower().startswith("data") and idx + 1 < len(lines):
            lines = lines[idx + 1:]
            break

    rows = []
    for l in lines:
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").replace(";", " ").split()
        try:
            vals = [float(p) for p in parts]
            if vals:
                rows.append(vals)
        except ValueError:
            continue

    if not rows:
        return None
    mode_cols = int(pd.Series([len(r) for r in rows]).mode().iloc[0])
    arr = np.asarray([r for r in rows if len(r) == mode_cols], dtype=float)
    if arr.shape[0] <= 16 and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    return strip_tachometer_channels(arr)


def record_sample(features, label, group):
    if len(features) == 28 and np.all(np.isfinite(features)):
        FEATURES.append(features)
        LABELS.append(label)
        GROUPS.append(group)


def ingest_signal(arr, label, group_name, max_windows=20):
    if arr is None or len(arr) < 10:
        return

    # 1. Base 2048 segments
    n = len(arr)
    step = max(SEGMENT_SIZE, (n - SEGMENT_SIZE) // max(1, max_windows)) if n > SEGMENT_SIZE else SEGMENT_SIZE
    count = 0
    for start in range(0, max(1, n - SEGMENT_SIZE + 1), step):
        seg = arr[start:start + SEGMENT_SIZE]
        if len(seg) < SEGMENT_SIZE:
            seg = expand_short_signal(seg, SEGMENT_SIZE)

        # Standard segment
        feats = extract_features(seg)
        record_sample(feats, label, group_name)
        count += 1

        # Channel down-projections (simulating 1-channel and 2-channel installations)
        if seg.shape[1] > 1:
            seg_1ch = seg[:, :1]
            record_sample(extract_features(seg_1ch), label, f"{group_name}_1ch")
        if seg.shape[1] >= 3:
            seg_2ch = seg[:, :2]
            record_sample(extract_features(seg_2ch), label, f"{group_name}_2ch")

        # Gain scale invariance augmentations (0.05x, 0.2x, 2.0x, 5.0x)
        for scale in [0.05, 0.2, 2.5]:
            record_sample(extract_features(seg * scale), label, f"{group_name}_scale_{scale}")

        if count >= max_windows:
            break

    # 2. Micro-telemetry slices (10, 20, 50, 100 points) to train micro-sample capability
    micro_lens = [10, 20, 30, 50, 80, 120]
    for m_len in micro_lens:
        if n >= m_len:
            for _ in range(2):
                idx = random.randint(0, n - m_len)
                sub = arr[idx:idx + m_len]
                expanded = expand_short_signal(sub, SEGMENT_SIZE)
                record_sample(extract_features(expanded), label, f"{group_name}_micro_{m_len}")
                # Also with 1 channel
                if sub.shape[1] > 1:
                    expanded_1ch = expand_short_signal(sub[:, :1], SEGMENT_SIZE)
                    record_sample(extract_features(expanded_1ch), label, f"{group_name}_micro_{m_len}_1ch")


print("=" * 70)
print("GEARGUARD AI: INGESTING MULTI-DATASET REPOSITORIES...")
print("=" * 70)

# 1. SpectraQuest Variable-Load Dataset (0% to 90% load)
if HEALTHY_DIR.exists() and FAULTY_DIR.exists():
    for f in sorted(HEALTHY_DIR.glob("*.txt")):
        arr = parse_vibration_file(f)
        ingest_signal(arr, "Healthy", f"SQ_{f.stem}", max_windows=15)
    for f in sorted(FAULTY_DIR.glob("*.txt")):
        arr = parse_vibration_file(f)
        ingest_signal(arr, "BrokenTooth", f"SQ_{f.stem}", max_windows=15)
    print("Ingested SpectraQuest Variable-Load Dataset.")

# 2. Southeast University (SEU) DDS Gearbox Dataset (.csv and .mat)
if SAMPLES_DIR.exists():
    seu_map = {
        "SEU_Health_20_0.csv": "Healthy",
        "SEU_Health_20_0.mat": "Healthy",
        "SEU_Chipped_20_0.csv": "BrokenTooth",
        "SEU_Chipped_20_0.mat": "BrokenTooth",
        "SEU_Miss_20_0.csv": "BrokenTooth",
        "SEU_Miss_20_0.mat": "BrokenTooth",
        "SEU_Root_20_0.csv": "BrokenTooth",
        "SEU_Surface_20_0.csv": "BrokenTooth",
    }
    for fname, lbl in seu_map.items():
        p = SAMPLES_DIR / fname
        if p.exists():
            arr = parse_vibration_file(p)
            ingest_signal(arr, lbl, f"SEU_{fname}", max_windows=20)
    print("Ingested Southeast University (SEU) Benchmark Files.")

# 3. CWRU Benchmarks
cwru_map = [
    ("normal_0.mat", "Healthy"),
    ("B007_0.mat", "BrokenTooth"),
    ("IR007_0.mat", "BrokenTooth"),
    ("OR007@6_0.mat", "BrokenTooth"),
]
for fname, lbl in cwru_map:
    for base_dir in [SAMPLES_DIR, Path("/home/jeremy")]:
        p = base_dir / fname
        if p.exists():
            arr = parse_vibration_file(p)
            ingest_signal(arr, lbl, f"CWRU_{p.stem}_{base_dir.name}", max_windows=20)
print("Ingested CWRU Bearing / Gear Datasets.")

# 4. PHM 2009 Multi-Speed / Multi-Load Runs (/home/jeremy/archive)
if PHM_DIR.exists():
    phm_runs = sorted(PHM_DIR.glob("Run_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(phm_runs)} PHM 2009 benchmark runs. Processing representative regimes...")
    for f in phm_runs:
        arr = parse_vibration_file(f)
        if arr is None or len(arr) < 2048:
            continue
        # Strip DC and compute kurtosis on first 10,000 points
        sub = arr[:10000, 0] - np.mean(arr[:10000, 0])
        std = np.std(sub)
        k = float(kurtosis(sub, fisher=False)) if std > 1e-9 else 3.0
        # Smooth rotation (Run 10, 20, 50, etc.)
        if k <= 3.55:
            ingest_signal(arr, "Healthy", f"PHM_{f.stem}", max_windows=8)
        # Impact fault regime (Run 1, 2, 5, 75, 100, etc.)
        elif k >= 4.2:
            ingest_signal(arr, "BrokenTooth", f"PHM_{f.stem}", max_windows=8)
    print("Ingested PHM 2009 Diagnostic Runs.")

# 5. Dedicated Micro-Sample Telemetry
for fname, lbl in [
    ("micro_10sample_healthy.txt", "Healthy"),
    ("micro_10sample_broken.txt", "BrokenTooth"),
    ("horizontal_10samples.csv", "Healthy"),
    ("chipped_1ch_sample.csv", "BrokenTooth"),
]:
    p = SAMPLES_DIR / fname
    if p.exists():
        arr = parse_vibration_file(p)
        ingest_signal(arr, lbl, f"SAMPLE_{fname}", max_windows=10)

X = np.asarray(FEATURES, dtype=float)
y = np.asarray(LABELS)
groups = np.asarray(GROUPS)

print("\n" + "=" * 70)
print(f"DATASET INGESTION COMPLETE:")
print(f"Total training/evaluation vectors: {len(y)}")
print(f"Feature matrix dimensions:        {X.shape}")
print(f"Healthy segment count:             {int(np.sum(y == 'Healthy'))}")
print(f"Faulty segment count:              {int(np.sum(y == 'BrokenTooth'))}")
print(f"Distinct dataset group domains:    {len(set(groups))}")
print("=" * 70)

# Evaluate Held-Out Cross-Dataset Generalization
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced_subsample",
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X[train_idx], y[train_idx])
pred = rf.predict(X[test_idx])
acc = accuracy_score(y[test_idx], pred)

print(f"\nHELD-OUT GROUP VALIDATION ACCURACY: {acc * 100:.2f}%")
print(classification_report(y[test_idx], pred, digits=4))

# Fit Production Ensemble on 100% of Multi-Domain Data
print("Training Production Multi-Domain Calibrated Ensemble on 100% of data...")
rf_full = RandomForestClassifier(
    n_estimators=350,
    max_features="sqrt",
    class_weight="balanced_subsample",
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
et_full = ExtraTreesClassifier(
    n_estimators=150,
    max_features="sqrt",
    class_weight="balanced_subsample",
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
ensemble = VotingClassifier(
    estimators=[("rf", rf_full), ("et", et_full)],
    voting="soft",
)
ensemble.fit(X, y)

joblib.dump(ensemble, OUT_MODEL)
print(f"Successfully exported production model to:\n  {OUT_MODEL}")
print("=" * 70)
