# Forest Fire Prediction

Predict the fire detection `confidence` score from `fire_archive.csv` using scikit‑learn (RandomForest, ExtraTrees). The project includes a Jupyter notebook for exploration and a Python script for quick runs.

## Files
- `Forest_Fires.ipynb` — EDA, preprocessing, modeling, and tuning
- `Forest_Fires_script.py` — Standalone pipeline (mirrors the notebook)
- `fire_archive.csv` — Input dataset

## Setup
```pwsh
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Run
```pwsh
python .\Forest_Fires_script.py
```

Notes: RF tuning uses RandomizedSearchCV for speed (RF single-threaded during search), then refits the best params at 300 trees with full parallelism.
