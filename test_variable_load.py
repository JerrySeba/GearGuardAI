
from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
HEALTHY_DIR = ROOT / "Healthy Data"
FAULTY_DIR = ROOT / "BrokenTooth Data"
SEGMENT_SIZE = 2048


def extract_features(segment):
    features = []

    for channel in range(4):
        x = segment[:, channel]
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))

        features.extend([
            np.mean(x),
            np.std(x),
            rms,
            peak,
            peak / (rms + 1e-12),
            pd.Series(x).kurt()
        ])

    return features


def load_recordings(folder, label):
    records = []

    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue

        try:
            data = pd.read_csv(
                file, sep=r"\s+", header=None, comment="#"
            ).apply(pd.to_numeric, errors="coerce")

            data = data.dropna(axis=0, how="any").to_numpy()

            if data.ndim != 2 or data.shape[1] != 4:
                print(f"Skipping unexpected format: {file.name}")
                continue

            if len(data) < SEGMENT_SIZE:
                continue

            # Example filename: h30hz30 or b30hz90
            match = re.search(r"hz(\d+)", file.stem.lower())
            if not match:
                print(f"Cannot read load from filename: {file.name}")
                continue

            load = int(match.group(1))

            for start in range(0, len(data) - SEGMENT_SIZE + 1,
                               SEGMENT_SIZE):
                segment = data[start:start + SEGMENT_SIZE]
                row = []

                for channel in range(4):
                    x = segment[:, channel]
                    rms = np.sqrt(np.mean(x ** 2))
                    peak = np.max(np.abs(x))

                    row.extend([
                        np.mean(x),
                        np.std(x),
                        rms,
                        peak,
                        peak / (rms + 1e-12),
                        pd.Series(x).kurt()
                    ])

                if np.all(np.isfinite(row)):
                    records.append({
                        "features": row,
                        "label": label,
                        "load": load,
                        "file": file.name
                    })

        except Exception as exc:
            print(f"Could not read {file.name}: {exc}")

    return records


if not HEALTHY_DIR.exists() or not FAULTY_DIR.exists():
    raise FileNotFoundError(
        "Expected Healthy Data and BrokenTooth Data beside this script."
    )

records = (
    load_recordings(HEALTHY_DIR, "Healthy")
    + load_recordings(FAULTY_DIR, "BrokenTooth")
)

if not records:
    raise RuntimeError("No usable recordings found.")

df = pd.DataFrame(records)
print("\nSegments by load and condition:")
print(pd.crosstab(df["load"], df["label"]))

loads = sorted(df["load"].unique())

if len(loads) < 2:
    raise RuntimeError(
        "Need at least two distinct load values for a load-held-out test."
    )

print("\nLEAVE-ONE-LOAD-OUT EVALUATION")

for held_out_load in loads:
    train = df[df["load"] != held_out_load]
    test = df[df["load"] == held_out_load]

    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        print(
            f"\nSkipping load {held_out_load}: "
            "training and test sets must each contain both classes."
        )
        continue

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(np.array(train["features"].tolist()), train["label"])
    predictions = model.predict(np.array(test["features"].tolist()))

    print(f"\nHeld-out load: {held_out_load}")
    print(f"Training recordings: {train['file'].nunique()}")
    print(f"Test recordings: {test['file'].nunique()}")
    print(f"Accuracy: {accuracy_score(test['label'], predictions) * 100:.2f}%")
    print(classification_report(
        test["label"], predictions, zero_division=0
    ))
