# trains a model that tells good reps from bad reps, using the reps you labeled yourself
# steps: export the reps csv from the page, fill the human_label column (good / bad),
# save it as backend/data/movement_reps.csv, then run:  python -m backend.train_movement
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import DATA_DIR, MODELS_DIR

NUMERIC = ["rep_duration_s", "min_angle", "max_angle", "rom", "max_lean", "min_symmetry"]
MIN_ROWS, MIN_PER_CLASS = 40, 10


# reads the csv files and keeps only the reps that have a good / bad label
def load_labeled(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
    df = df[df["human_label"].isin(["good", "bad"])].reset_index(drop=True)
    return df


# trains and compares a few models, saves the best one
def run(paths=None, models_dir=MODELS_DIR):
    paths = paths or [DATA_DIR / "movement_reps.csv"]
    df = load_labeled(paths)
    counts = df["human_label"].value_counts()
    if len(df) < MIN_ROWS or counts.min() < MIN_PER_CLASS or len(counts) < 2:
        raise SystemExit(f"Not enough labeled reps: {counts.to_dict()} (need >= {MIN_ROWS} total and "
                         f">= {MIN_PER_CLASS} of each class). Label more reps.")
    X = pd.get_dummies(df[NUMERIC + ["exercise"]], columns=["exercise"], dtype=float)
    y = (df["human_label"] == "bad").astype(int)
    groups = df["session_id"]
    if groups.nunique() >= 3:
        cv, kw, how = GroupKFold(n_splits=min(5, groups.nunique())), {"groups": groups}, "GroupKFold by video/session"
    else:
        cv, kw, how = StratifiedKFold(5, shuffle=True, random_state=42), {}, \
            "StratifiedKFold (less than 3 videos, so reps of the same video can land in both parts, record more videos)"

    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42),
    }
    report = {"labeled_reps": len(df), "class_counts": counts.to_dict(), "cv": how,
              "hand_written_rules_f1": round(float(f1_score(y, (df["heuristic_label"] == "bad").astype(int))), 3)}
    for name, m in models.items():
        pred = cross_val_predict(m, X, y, cv=cv, **kw)
        report[name + "_f1_bad_class"] = round(float(f1_score(y, pred)), 3)

    best = max(("logreg", "random_forest"), key=lambda n: report[n + "_f1_bad_class"])
    models[best].fit(X, y)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": models[best], "columns": list(X.columns), "name": best, "report": report},
                models_dir / "movement_model.pkl")
    print(report)
    return report


if __name__ == "__main__":
    run([Path(a) for a in sys.argv[1:]] or None)
