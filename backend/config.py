from pathlib import Path

# main project folder (one level above backend/)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "backend" / "data"
MODELS_DIR = BASE_DIR / "backend" / "models"
FRONTEND_INDEX = BASE_DIR / "frontend" / "index.html"

REAL_CSV_PATH = DATA_DIR / "gym_members_clean.csv"
MODELS_PATH = MODELS_DIR / "smart_gym_models.pkl"

# the 10 columns the models are trained on
FEATURE_COLS = [
    "Age", "Weight_kg", "Height_m", "Max_BPM", "Avg_BPM", "Resting_BPM",
    "Session_Duration_min", "Fat_Percentage", "Water_Intake_liters", "BMI",
]

# the columns the risk rule depends on (all of them must come from the user)
PHYSIO_KEYS = ("Max_BPM", "Avg_BPM", "Resting_BPM", "Water_Intake_liters")
