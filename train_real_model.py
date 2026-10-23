from pathlib import Path
import io
import numpy as np
import pandas as pd
import scipy.io as sio
import joblib

from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
HEALTHY_DIR = ROOT / "Healthy Data"
FAULTY_DIR = ROOT / "BrokenTooth Data"
PHM_DIR = Path("/home/jeremy/archive")
SAMPLES_DIR = ROOT / "samples"

SEGMENT_SIZE = 2048
FEATURES = []
LABELS = []
GROUPS = []


def strip_tachometer_channels(arr):
    """
    Removes 0-5V square-wave tachometer pulse channels (such as Channel 3 in PHM 2009)
    so only true vibration accelerometer channels are analyzed.
    """
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
        # A 0-5V TTL tachometer square wave has mean > 1.0, ptp > 3.0, and Pearson kurtosis ~ 1.0 (< 1.45)
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
        variances = [float(np.var(segment[:, i] - np.mean(segment[:, i]))) for i in range(C)]
        return list(np.argsort(variances)[-target_channels:][::-1])


def extract_features(segment):
    segment = strip_tachometer_channels(segment)
    ch_map = adapt_channel_projection(segment, target_channels=4)
    result = []

    for channel in ch_map:
        raw_x = segment[:, channel].astype(float)
        # Always remove DC offset (critical for PHM 2009 & SEU sensors)
        x = raw_x - np.mean(raw_x)
        rms = float(np.sqrt(np.mean(x ** 2)))
        std = float(np.std(x)) if len(x) > 1 else 0.0
        mean_abs = float(np.mean(np.abs(x)))
        peak = float(np.max(np.abs(x)))

        kurt_val = float(kurtosis(x, fisher=False)) if len(x) >= 4 and std > 1e-9 else 3.0
        skew_val = float(skew(x)) if len(x) >= 3 and std > 1e-9 else 0.0
        if not np.isfinite(kurt_val):
            kurt_val = 3.0
        if not np.isfinite(skew_val):
            skew_val = 0.0

        result.extend([
            rms,
            std,
            mean_abs,
            peak,
            peak / (rms + 1e-12),
            kurt_val,
            skew_val,
        ])

    return result


def parse_any_file(file_path):
    raw = file_path.read_bytes()
    if file_path.suffix.lower() == ".mat" or raw[:6] == b"MATLAB":
        mat = sio.loadmat(io.BytesIO(raw))
        arrays_2d = []
        vectors_1d = []
        for k in sorted(mat.keys()):
            if k.startswith("__"):
                continue
            val = np.asarray(mat[k])
            if not np.issubdtype(val.dtype, np.number) or val.size <= 4:
                continue
            val = np.squeeze(val)
            if val.ndim == 1 and val.size >= 4:
                vectors_1d.append(val.astype(float))
            elif val.ndim == 2 and val.size >= 8:
                if val.shape[0] < val.shape[1] and val.shape[0] <= 16:
                    val = val.T
                arrays_2d.append(val.astype(float))
        if arrays_2d:
            arr = max(arrays_2d, key=lambda a: a.size)
            return strip_tachometer_channels(arr)
        elif vectors_1d:
            min_l = min(len(v) for v in vectors_1d)
            arr = np.column_stack([v[:min_l] for v in vectors_1d])
            return strip_tachometer_channels(arr)

    text = raw.decode("utf-8", errors="ignore")
    rows = []
    for line in text.splitlines():
        s = line.strip()
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
    col_counts = pd.Series([len(r) for r in rows]).mode().iloc[0]
    clean_rows = [r for r in rows if len(r) == col_counts]
    arr = np.asarray(clean_rows, dtype=float)
    if arr.shape[0] <= 16 and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    return strip_tachometer_channels(arr)


def add_segments_from_array(arr, label, group_name, max_segments=35):
    if arr is None or len(arr) < SEGMENT_SIZE:
        return
    count = 0
    step = max(SEGMENT_SIZE, (len(arr) - SEGMENT_SIZE) // max(1, max_segments))
    for start in range(0, len(arr) - SEGMENT_SIZE + 1, step):
        seg = arr[start:start + SEGMENT_SIZE]
        feats = extract_features(seg)
        if np.all(np.isfinite(feats)):
            FEATURES.append(feats)
            LABELS.append(label)
            GROUPS.append(group_name)
            count += 1
            if count >= max_segments:
                break


# 1. Load SpectraQuest Dataset
if HEALTHY_DIR.exists() and FAULTY_DIR.exists():
    for f in sorted(HEALTHY_DIR.glob("*.txt")):
        add_segments_from_array(parse_any_file(f), "Healthy", f"SQ_{f.name}", max_segments=25)
    for f in sorted(FAULTY_DIR.glob("*.txt")):
        add_segments_from_array(parse_any_file(f), "BrokenTooth", f"SQ_{f.name}", max_segments=25)

# 2. Load PHM 2009 Dataset (/home/jeremy/archive)
if PHM_DIR.exists():
    for f in sorted(PHM_DIR.glob("Run_*.csv"), key=lambda p: int(p.stem.split("_")[1])):
        arr = parse_any_file(f)
        if arr is None or len(arr) < SEGMENT_SIZE:
            continue
        # Evaluate physical impact kurtosis on AC-coupled channels
        x0 = arr[:10000, 0] - np.mean(arr[:10000, 0])
        x1 = arr[:10000, 1] - np.mean(arr[:10000, 1]) if arr.shape[1] > 1 else x0
        k_max = max(float(kurtosis(x0, fisher=False)), float(kurtosis(x1, fisher=False)))
        if k_max <= 3.55:
            add_segments_from_array(arr, "Healthy", f"PHM_{f.name}", max_segments=8)
        elif k_max >= 4.25:
            add_segments_from_array(arr, "BrokenTooth", f"PHM_{f.name}", max_segments=5)

# 3. Load Southeast University (SEU) & Mechanical-datasets (.mat & .csv)
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
        "normal_0.mat": "Healthy",
        "B007_0.mat": "BrokenTooth",
        "IR007_0.mat": "BrokenTooth",
    }
    for fname, lbl in seu_map.items():
        fpath = SAMPLES_DIR / fname
        if fpath.exists():
            arr = parse_any_file(fpath)
            # Tile if sample chunk is compact so multiple training windows are represented
            if arr is not None and len(arr) < SEGMENT_SIZE * 10:
                arr = np.tile(arr, (10, 1))
            add_segments_from_array(arr, lbl, f"BENCH_{fname}", max_segments=20)

# Also check full /home/jeremy/*.mat files
for fname, lbl in [("normal_0.mat", "Healthy"), ("B007_0.mat", "BrokenTooth"), ("IR007_0.mat", "BrokenTooth")]:
    fpath = Path("/home/jeremy") / fname
    if fpath.exists():
        add_segments_from_array(parse_any_file(fpath), lbl, f"MAT_{fname}", max_segments=25)

X = np.asarray(FEATURES, dtype=float)
y = np.asarray(LABELS)
groups = np.asarray(GROUPS)

print("Total multi-domain segments loaded:", len(y))
print("Feature matrix:", X.shape)
print("Healthy segments:", int(np.sum(y == "Healthy")))
print("Faulty segments:", int(np.sum(y == "BrokenTooth")))

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X[train_idx], y[train_idx])

pred = model.predict(X[test_idx])
print("\nHeld-Out Group Test Accuracy:", round(accuracy_score(y[test_idx], pred) * 100, 2), "%")
print(classification_report(y[test_idx], pred))

# Refit on 100% of multi-domain data for production deployment
model.fit(X, y)
out_file = ROOT / "gearguard_real_model.joblib"
joblib.dump(model, out_file)
print(f"Saved multi-domain model to: {out_file}")
