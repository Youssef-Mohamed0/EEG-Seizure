# SmartGym AI

FastAPI + scikit-learn + MediaPipe Pose (runs in the browser).

## Run
    pip install -r requirements.txt
    python -m backend.ml_pipeline          # trains models and prints the metrics
    uvicorn backend.api:app --reload       # open http://127.0.0.1:8000

## What is real / what is not
- Data: Kaggle Gym Members Exercise Dataset (973 rows). FitBit data is NOT merged
  (raw copy kept in backend/data/raw/ only). `archive/` holds an unused synthetic CSV.
- `Overtraining_Risk_Label` is a **rule** over the input features (age-predicted max HR,
  hydration, resting HR). The classifier re-learns that rule: a depth-3 decision tree gets
  F1 0.913 vs RandomForest 0.940 (majority-class baseline 0.789). It is not validated
  injury prediction.
- Calories regression: LinearRegression R2 0.9643 vs RandomForest 0.9649 - same quality.
- Movement risk = transparent heuristic on MediaPipe numbers (thresholds not learned).
- Physiological risk is only reported when the user supplies heart-rate + water data.
  Nothing is estimated from rep counts anymore.
- Full numbers: printed by `python -m backend.ml_pipeline` (also stored inside the .pkl)

## Building a real movement dataset (step 3)
1. In the web page, play a video; every completed rep is logged. Click **Export reps CSV**.
2. Open the CSV, watch the video (use `video_time_s`), fill `human_label` = good / bad.
3. Save/merge into `backend/data/movement_reps.csv` (aim for 100+ reps, 3+ different videos).
4. `python -m backend.train_movement` -> cross-validated F1 vs the hand-written rules.

## Team workflow
- `notebooks/` = ML team experiments (one notebook per member). `pip install -r requirements-notebooks.txt`, then `jupyter lab`.
- Every notebook starts with the same cell: `load_data_and_split()` = one official split (seed 42), so all results are comparable.
- Members edit ONLY their notebook, on their own branch. Backend/frontend changes go through the team lead (pull request).
