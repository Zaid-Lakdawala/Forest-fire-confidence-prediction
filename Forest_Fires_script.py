"""
Forest Fire Prediction - Script Version

This script mirrors the current `Forest_Fires.ipynb` notebook workflow: imports, data loading,
EDA, preprocessing/feature engineering, train/test split, baseline RandomForestRegressor,
RandomForest GridSearchCV tuning, ExtraTrees baseline, and ExtraTrees RandomizedSearchCV tuning.
"""

import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


# Load data and quick checks
forest = pd.read_csv('fire_archive.csv')
print(f"Data shape: {forest.shape}")
print(f"Missing values total: {forest.isnull().sum().sum()}")
print("Head:\n", forest.head())

# EDA: describe + correlations heatmap
print("Describe:\n", forest.describe())
plt.figure(figsize=(10, 10))
numeric_forest = forest.select_dtypes(include=[np.number])
corr = numeric_forest.corr()
sns.heatmap(corr, annot=True, cmap='viridis', linewidths=.5)
plt.title('Correlation Heatmap (All Numeric)')
plt.tight_layout()
plt.show()

# Categorical summaries
print("The scan column")
print(forest['scan'].value_counts())
print()
print("The aqc_time column")
print(forest['acq_time'].value_counts())
print()
print("The satellite column")
print(forest['satellite'].value_counts())
print()
print("The instrument column")
print(forest['instrument'].value_counts())
print()
print("The version column")
print(forest['version'].value_counts())
print()
print("The daynight column")
print(forest['daynight'].value_counts())

# Preprocessing: drops, maps, dummies, bins, dates
forest = forest.drop(['track'], axis=1)

forest = forest.drop(['instrument', 'version'], axis=1)

daynight_map = {"D": 1, "N": 0}
satellite_map = {"Terra": 1, "Aqua": 0}
forest['daynight'] = forest['daynight'].map(daynight_map)
forest['satellite'] = forest['satellite'].map(satellite_map)

types = pd.get_dummies(forest['type'])
forest = pd.concat([forest, types], axis=1)
forest = forest.drop(['type'], axis=1)
forest = forest.rename(columns={0: 'type_0', 2: 'type_2', 3: 'type_3'})

bins = [0, 1, 2, 3, 4, 5]
labels = [1, 2, 3, 4, 5]
forest['scan_binned'] = pd.cut(forest['scan'], bins=bins, labels=labels)

forest['acq_date'] = pd.to_datetime(forest['acq_date'])
forest = forest.drop(['scan'], axis=1)
forest['year'] = forest['acq_date'].dt.year
forest['month'] = forest['acq_date'].dt.month
forest['day'] = forest['acq_date'].dt.day

print("Post-preprocessing shape:", forest.shape)

# Train/test split
y = forest['confidence']
fin = forest.drop(['confidence', 'acq_date', 'acq_time', 'bright_t31', 'type_0'], axis=1)
Xtrain, Xtest, ytrain, ytest = train_test_split(fin, y, test_size=0.2, random_state=42)

# Baseline RandomForestRegressor
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(Xtrain, ytrain)
print('RF R^2 train:', rf.score(Xtrain, ytrain))
print('RF R^2 test :', rf.score(Xtest, ytest))

# RandomForestRegressor hyperparameter tuning with RandomizedSearchCV (faster)
rf_param_dist = {
    'n_estimators': [50, 100, 150],  # fewer trees during search
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Avoid double parallelism: RF single-threaded during search, CV uses n_jobs=-1
rf_base = RandomForestRegressor(random_state=42, n_jobs=1)
rf_rs = RandomizedSearchCV(rf_base, rf_param_dist, n_iter=20, cv=2, random_state=42, n_jobs=-1, verbose=2)
rf_rs.fit(Xtrain, ytrain)
print('Best RF params (RandomizedSearchCV):', rf_rs.best_params_)

# Refit best model with more trees and full parallelism
best_params = rf_rs.best_params_.copy()
best_params['n_estimators'] = 300
rf_final = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
rf_final.fit(Xtrain, ytrain)
print('Tuned RF R^2 test (RandomizedSearchCV + refit 300):', rf_final.score(Xtest, ytest))

# ExtraTreesRegressor baseline and metrics
extra = ExtraTreesRegressor(n_estimators=400, max_features='sqrt', random_state=42, n_jobs=-1)
extra.fit(Xtrain, ytrain)

y_train_pred = extra.predict(Xtrain)
y_test_pred = extra.predict(Xtest)
print('ExtraTrees R^2 train:', r2_score(ytrain, y_train_pred))
print('ExtraTrees R^2 test :', r2_score(ytest, y_test_pred))
print('ExtraTrees MAE      :', mean_absolute_error(ytest, y_test_pred))

# Quick hyperparameter search for ExtraTrees (small for speed)
param_dist = {
    'n_estimators': [200, 300, 400, 600],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 20, 28, 32],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4],
}

et_base = ExtraTreesRegressor(random_state=42, n_jobs=-1)
et_search = RandomizedSearchCV(et_base, param_dist, n_iter=15, cv=2, random_state=42, n_jobs=-1, verbose=1)
et_search.fit(Xtrain, ytrain)
print('Best ExtraTrees params:', et_search.best_params_)

extra_tuned = ExtraTreesRegressor(**et_search.best_params_, random_state=42, n_jobs=-1)
extra_tuned.fit(Xtrain, ytrain)
print('Tuned ExtraTrees R^2 test:', extra_tuned.score(Xtest, ytest))
