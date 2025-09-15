# Forest Fire Prediction

This project builds regression models to predict the `confidence` score of fire detections from the "fire_archive.csv" dataset. It includes a Jupyter notebook (`Forest_Fires.ipynb`) and a mirrored Python script (`Forest_Fires_script.py`).

## Contents

- `Forest_Fires.ipynb`: Exploratory data analysis (EDA), preprocessing, feature engineering, and modeling.
- `Forest_Fires_script.py`: Standalone script replicating the notebook's pipeline.
- `fire_archive.csv`: Dataset used for training and evaluation.

## Environment Setup

Use Python 3.9+ and install required packages:

```pwsh
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy seaborn matplotlib scikit-learn
```

If you plan to try gradient boosting variants (not in the script by default):

```pwsh
pip install lightgbm xgboost
```

## Data Overview

Key columns used:

- Categorical: `satellite` (mapped Aqua/Terra to 0/1), `daynight` (N/D to 0/1), `type` (one-hot encoded into `type_0`, `type_2`, `type_3`).
- Date: `acq_date` (extracted year, month, day), `acq_time` (dropped in baseline pipeline).
- Numeric: various brightness and geospatial features; `scan` is binned and original is dropped.

Dropped columns: `track`, `instrument`, `version`, `scan`, `acq_time`, `bright_t31` (per current pipeline), and `type_0` (to avoid dummy trap).

Target: `confidence` (regression).

## What the Script Does

1. Loads `fire_archive.csv`, prints basic info and head.
2. EDA: correlation heatmap of numeric fields.
3. Preprocessing:
   - Drops unused columns (`track`, `instrument`, `version`).
   - Maps `daynight` and `satellite` to numeric values.
   - One-hot encodes `type` and renames columns.
   - Bins `scan` into `scan_binned` and drops original `scan`.
   - Parses `acq_date` and extracts `year`, `month`, `day`.
4. Train/test split on processed features.
5. Models:
   - Baseline RandomForestRegressor (300 trees).
   - RandomForest tuning via RandomizedSearchCV (single-threaded RF during search to avoid double parallelism). Best params are refit at 300 trees with full parallelism.
   - ExtraTrees baseline and metrics (R², MAE).
   - ExtraTrees tuning via RandomizedSearchCV (compact search).

## How to Run

Run the script from the project folder:

```pwsh
.venv\Scripts\Activate.ps1
python .\Forest_Fires_script.py
```

The script will print data info, show a correlation heatmap, and then print training/testing scores and best hyperparameters. Plots open in a window; close the window to continue execution.

## Notes on Performance

- ## Notes on Performance
- The script uses RandomizedSearchCV for RandomForest for faster tuning and sets the estimator to `n_jobs=1` during search to avoid over-subscription. The best params are then refit at `n_estimators=300` with `n_jobs=-1`.
- If runtime is still long, reduce `n_iter`, shrink the search space, or use `max_samples` in the estimator (e.g., `max_samples=0.7`) during search.
- Set `n_jobs=-1` to use all CPU cores for the final refit phase.

## Reproducibility

- Random seeds are set with `random_state=42` for comparable runs.
