"""Single source of truth for paths and feature names (paths no longer depend on the cwd)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SMARTGYM_DATA_DIR", BASE_DIR / "backend" / "data"))
MODELS_DIR = Path(os.getenv("SMARTGYM_MODELS_DIR", BASE_DIR / "backend" / "models"))
FRONTEND_INDEX = BASE_DIR / "frontend" / "index.html"

# Kaggle "Gym Members Exercise Dataset" (973 rows), cleaned. FitBit data is NOT merged in.
REAL_CSV_PATH = DATA_DIR / "gym_members_clean.csv"
MODELS_PATH = MODELS_DIR / "smart_gym_models.pkl"

FEATURE_COLS = [
    "Age", "Weight_kg", "Height_m", "Max_BPM", "Avg_BPM", "Resting_BPM",
    "Session_Duration_min", "Fat_Percentage", "Water_Intake_liters", "BMI",
]
# Inputs the label rule depends on. If any of these is not supplied by the user,
# the physiological risk is NOT reported (we refuse to guess it).
PHYSIO_KEYS = ("Max_BPM", "Avg_BPM", "Resting_BPM", "Water_Intake_liters")
