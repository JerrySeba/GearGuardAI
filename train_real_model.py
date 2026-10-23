
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

# Dataset folders should be beside this Python file.
ROOT = Path(__file__).resolve().parent
HEALTHY_DIR = ROOT / "Healthy Data"
FAULTY_DIR = ROOT / "BrokenTooth Data"

SEGMENT_SIZE = 2048
FEATURES = []
LABELS = []
GROUPS = []

def extract_features(segment):
    result = []

    for channel in range(segment.shape[1]):
        x = segment[:, channel].astype(float)
        rms = np.sqrt(np.mean(x ** 2))
        std = np.std(x)
        mean_abs = np.mean(np.abs(x))
        peak = np.max(np.abs(x))

        result.extend([
            rms,
            std,
            mean_abs,
            peak,
            peak / (rms + 1e-12),
            kurtosis(x, fisher=False),
            skew(x),
        ])

    return result

def load_folder(folder, label):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(p for p in folder.iterdir() if p.is_file())
    if not files:
        raise RuntimeError(f"No data files found in {folder}")

    for file in files:
        print(f"Reading {file.name} ...")

        # Whitespace-separated values; no header assumed.
        data = pd.read_csv(
            file, sep=r"\s+", header=None, comment="#"
        ).apply(pd.to_numeric, errors="coerce")

        data = data.dropna(axis=0, how="any").to_numpy()

        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(
                f"{file.name}: expected 4 numeric columns; "
                f"found shape {data.shape}. Inspect this file format."
            )

        if len(data) < SEGMENT_SIZE:
            print(f"Skipping short file: {file.name}")
            continue

        # Non-overlapping segments keep processing manageable.
        for start in range(0, len(data) - SEGMENT_SIZE + 1,
                           SEGMENT_SIZE):
            segment = data[start:start + SEGMENT_SIZE]
            features = extract_features(segment)

            if np.all(np.isfinite(features)):
                FEATURES.append(features)
                LABELS.append(label)
                # Group by original file to prevent segment leakage.
                GROUPS.append(file.name)

load_folder(HEALTHY_DIR, "Healthy")
load_folder(FAULTY_DIR, "BrokenTooth")

X = np.asarray(FEATURES, dtype=float)
y = np.asarray(LABELS)
groups = np.asarray(GROUPS)

print("\nSegments loaded:", len(y))
print("Feature matrix:", X.shape)
print("Healthy segments:", np.sum(y == "Healthy"))
print("Broken-tooth segments:", np.sum(y == "BrokenTooth"))

# Hold out complete recordings rather than random segments.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups))

if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
    raise RuntimeError(
        "The file-level split does not contain both classes in train "
        "and test. We need to adjust the split strategy."
    )

model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42,
    n_jobs=-1
)
model.fit(X[train_idx], y[train_idx])

pred = model.predict(X[test_idx])

print("\nGEARGUARD AI — REAL DATA BASELINE")
print("Held-out-file accuracy:",
      round(accuracy_score(y[test_idx], pred) * 100, 2), "%")
print(classification_report(y[test_idx], pred, zero_division=0))

joblib.dump(model, ROOT / "gearguard_real_model.joblib")
print("Saved model: gearguard_real_model.joblib")
print("Note: this baseline classifies healthy vs broken-tooth.")