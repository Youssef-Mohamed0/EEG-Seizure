"""Training pipeline (leak-free order: split FIRST, then fit every transformer on train only).

Run from the project root:   python -m backend.ml_pipeline
Writes: backend/models/smart_gym_models.pkl (metrics are stored inside it and printed)

NOTE: Overtraining_Risk_Label is a deterministic RULE over input features (see
build_rule_label). The classifier therefore re-learns that rule; its score measures rule
re-discovery, not real-world injury prediction. Baselines are reported for that reason.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backend.config import FEATURE_COLS, MODELS_DIR, MODELS_PATH, REAL_CSV_PATH
from backend.cluster_naming import describe_clusters
from backend.features import enhance_features

SEED = 42


def build_rule_label(df: pd.DataFrame) -> pd.Series:
    """Sports-medicine style rule: >96% of age-predicted max HR, or dehydrated + high avg HR,
    or high resting HR + long session."""
    ratio = df["Max_BPM"] / (220.0 - df["Age"])
    return ((ratio > 0.96)
            | ((df["Avg_BPM"] > 155) & (df["Water_Intake_liters"] < 2.0))
            | ((df["Resting_BPM"] > 82) & (df["Session_Duration_min"] > 85))).astype(int)


def download_gym_dataset(dest=REAL_CSV_PATH) -> pd.DataFrame:
    """One-off: download Kaggle gym-members data and write the cleaned CSV. (FitBit is not used.)"""
    import glob, os
    import kagglehub  # lazy import: only needed when the CSV does not exist yet
    path = kagglehub.dataset_download("valakhorasani/gym-members-exercise-dataset")
    g = pd.read_csv(glob.glob(os.path.join(path, "*.csv"))[0])
    df = pd.DataFrame({
        "Age": g["Age"].astype(float), "Weight_kg": g["Weight (kg)"].astype(float),
        "Height_m": g["Height (m)"].astype(float), "Max_BPM": g["Max_BPM"].astype(float),
        "Avg_BPM": g["Avg_BPM"].astype(float), "Resting_BPM": g["Resting_BPM"].astype(float),
        "Session_Duration_min": (g["Session_Duration (hours)"].astype(float) * 60).round(1),
        "Fat_Percentage": g["Fat_Percentage"].astype(float),
        "Water_Intake_liters": g["Water_Intake (liters)"].astype(float),
        "BMI": g["BMI"].astype(float), "Workout_Type": g["Workout_Type"].astype(str),
        "Experience_Level": g["Experience_Level"].astype(int),
        "Calories_Burned": g["Calories_Burned"].astype(float),
    }).dropna().reset_index(drop=True)
    df["Overtraining_Risk_Label"] = build_rule_label(df)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def _rf_clf():
    return RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=SEED)


def load_data_and_split(csv_path=REAL_CSV_PATH):
    """THE official train/test split (seed 42, stratified, 80/20). Notebooks must use this too,
    so the whole team evaluates on exactly the same 195 test rows.
    Returns: df, X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test"""
    path = Path(csv_path)
    df = pd.read_csv(path) if path.exists() else download_gym_dataset(path)
    X, y_c, y_r = df[FEATURE_COLS], df["Overtraining_Risk_Label"], df["Calories_Burned"]
    Xtr, Xte, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_c, y_r, test_size=0.2, random_state=SEED, stratify=y_c)
    return df, Xtr, Xte, yc_tr, yc_te, yr_tr, yr_te


def run_real_ml_pipeline(csv_path=REAL_CSV_PATH, models_dir=MODELS_DIR):
    # 1) split FIRST (every transformer below is fitted on the TRAIN part only)
    df, Xtr, Xte, yc_tr, yc_te, yr_tr, yr_te = load_data_and_split(csv_path)

    # 2) unsupervised stages, fitted on TRAIN only
    scaler = StandardScaler().fit(Xtr)
    Xs_tr = scaler.transform(Xtr)
    pca = PCA(n_components=3, random_state=SEED).fit(Xs_tr)
    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(Xs_tr)
    iso = IsolationForest(contamination=0.06, random_state=SEED).fit(Xs_tr)
    sil_by_k = {k: round(float(silhouette_score(Xs_tr, KMeans(k, random_state=SEED, n_init=10).fit_predict(Xs_tr))), 3)
                for k in range(2, 7)}
    db = DBSCAN(eps=2.1, min_samples=8).fit_predict(Xs_tr)  # diagnostic only, not used at inference

    cluster_profiles = describe_clusters(kmeans.cluster_centers_, FEATURE_COLS, np.bincount(kmeans.labels_))
    Etr, _, _ = enhance_features(Xtr, scaler, pca, kmeans)
    Ete, _, _ = enhance_features(Xte, scaler, pca, kmeans)

    # 3) supervised models + honest baselines
    clf = _rf_clf().fit(Etr, yc_tr)
    reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=SEED).fit(Etr, yr_tr)
    clf_raw = _rf_clf().fit(Xtr, yc_tr)
    dt3 = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr, yc_tr)
    dummy_c = DummyClassifier(strategy="most_frequent").fit(Xtr, yc_tr)
    lin = LinearRegression().fit(Xtr, yr_tr)
    dummy_r = DummyRegressor().fit(Xtr, yr_tr)

    def c_metrics(model, A):
        return {"f1": round(float(f1_score(yc_te, model.predict(A))), 4)}
    p = clf.predict_proba(Ete)[:, 1]
    rmse = float(np.sqrt(mean_squared_error(yr_te, reg.predict(Ete))))
    metrics = {
        "rows": int(len(df)), "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
        "label_positive_rate": round(float(df["Overtraining_Risk_Label"].mean()), 3),
        "pca_variance_pct_3comp": round(float(pca.explained_variance_ratio_.sum() * 100), 2),
        "silhouette_by_k_train": sil_by_k,
        "cluster_profiles": cluster_profiles,
        "dbscan_noise_points_train": int((db == -1).sum()),
        "dbscan_clusters_train": int(len(set(db)) - (1 if -1 in db else 0)),
        "isolation_forest_anomalies_train": int((iso.predict(Xs_tr) == -1).sum()),
        "classifier": {
            "rf_13_features": {**c_metrics(clf, Ete), "roc_auc": round(float(roc_auc_score(yc_te, p)), 4)},
            "rf_10_raw_features": c_metrics(clf_raw, Xte),
            "decision_tree_depth3": {**c_metrics(dt3, Xte), "roc_auc": round(float(roc_auc_score(yc_te, dt3.predict_proba(Xte)[:, 1])), 4)},
            "majority_class": c_metrics(dummy_c, Xte),
        },
        "regressor": {
            "rf_13_features": {"r2": round(float(r2_score(yr_te, reg.predict(Ete))), 4),
                               "mae": round(float(mean_absolute_error(yr_te, reg.predict(Ete))), 2),
                               "rmse": round(rmse, 2)},
            "linear_regression_10": {"r2": round(float(r2_score(yr_te, lin.predict(Xte))), 4),
                                     "mae": round(float(mean_absolute_error(yr_te, lin.predict(Xte))), 2)},
            "mean_baseline": {"r2": round(float(r2_score(yr_te, dummy_r.predict(Xte))), 4),
                              "mae": round(float(mean_absolute_error(yr_te, dummy_r.predict(Xte))), 2)},
        },
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "scaler": scaler, "pca": pca, "kmeans": kmeans, "iso_forest": iso,
        "classifier": clf, "regressor": reg, "feature_cols": FEATURE_COLS,
        "cluster_profiles": cluster_profiles,
        "training_medians": {k: float(v) for k, v in Xtr.median().items()},
        "metrics_summary": metrics,
    }, models_dir / MODELS_PATH.name)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run_real_ml_pipeline()
