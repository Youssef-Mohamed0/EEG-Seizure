import os
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from backend.config import FEATURE_COLS, FRONTEND_INDEX, REAL_CSV_PATH
from backend.pose_analyzer import SmartGymPoseAnalyzer

app = FastAPI(title="SmartGym AI")

# Frontend is served from the same origin, so CORS is only needed for dev from another origin.
_origins = [o for o in os.getenv("SMARTGYM_CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@lru_cache(maxsize=1)
def get_analyzer() -> SmartGymPoseAnalyzer:
    return SmartGymPoseAnalyzer()


class BiometricPoseInput(BaseModel):
    # required body profile
    Age: float = Field(ge=10, le=100)
    Weight_kg: float = Field(ge=30, le=250)
    Height_m: float = Field(ge=1.2, le=2.3)
    Session_Duration_min: float = Field(ge=1, le=300)
    # optional smartwatch data: if omitted the physiological risk is NOT reported
    Max_BPM: Optional[float] = Field(default=None, ge=80, le=230)
    Avg_BPM: Optional[float] = Field(default=None, ge=50, le=220)
    Resting_BPM: Optional[float] = Field(default=None, ge=30, le=120)
    Fat_Percentage: Optional[float] = Field(default=None, ge=2, le=60)
    Water_Intake_liters: Optional[float] = Field(default=None, ge=0, le=10)
    # camera-derived (MediaPipe) numbers
    torso_lean_deg: float = Field(default=0.0, ge=0, le=90)
    symmetry_score: float = Field(default=100.0, ge=0, le=100)
    good_reps: int = Field(default=0, ge=0, le=1000)
    bad_reps: int = Field(default=0, ge=0, le=1000)

    @model_validator(mode="after")
    def _hr_consistent(self):
        if self.Max_BPM is not None and self.Avg_BPM is not None and self.Avg_BPM > self.Max_BPM:
            raise ValueError("Avg_BPM cannot exceed Max_BPM")
        return self


@app.get("/")
def serve_dashboard():
    return FileResponse(FRONTEND_INDEX)


@lru_cache(maxsize=1)
def _overview() -> dict:
    """Computed once (vectorised), then served from cache."""
    a = get_analyzer()
    df = pd.read_csv(REAL_CSV_PATH)
    Xs = a.scaler.transform(df[FEATURE_COLS])
    Xp = a.pca.transform(Xs)
    clusters = a.kmeans.predict(Xs)
    anom = a.iso_forest.predict(Xs) == -1
    wt = df["Workout_Type"] if "Workout_Type" in df else pd.Series(["Gym"] * len(df))
    points = [{"pca1": round(float(p1), 3), "pca2": round(float(p2), 3), "cluster": int(c),
               "is_anomaly": bool(an), "calories": float(cal), "workout_type": str(w)}
              for p1, p2, c, an, cal, w in zip(Xp[:, 0], Xp[:, 1], clusters, anom, df["Calories_Burned"], wt)]
    return {
        "dataset_source": "Kaggle Gym Members Exercise Dataset (973 rows). FitBit data is not merged.",
        "total_records": int(len(df)),
        "features_count": len(FEATURE_COLS),
        "pca_explained_variance": round(float(a.pca.explained_variance_ratio_.sum() * 100), 2),
        "pca_explained_variance_2d": round(float(a.pca.explained_variance_ratio_[:2].sum() * 100), 2),
        "total_anomalies": int(anom.sum()),
        "cluster_names": {str(k): {"en": v["en"], "ar": v["ar"], "size": v["size"]} for k, v in a.cluster_profiles.items()},
        "metrics": a.metrics_summary,
        "scatter_data": points,
    }


@app.get("/api/dataset-overview")
def get_dataset_overview():
    return _overview()


@app.post("/api/analyze-biometrics")
def analyze_biometrics(data: BiometricPoseInput):
    payload = data.model_dump()          # inputs are passed through untouched
    return {"input_metrics": payload, "ml_results": get_analyzer().evaluate_real_biometrics_and_pose(payload)}
