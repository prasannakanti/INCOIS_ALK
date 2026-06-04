### Joshi_et_al_2025_TA_NIO
# Joshi, A. P., Ghoshal, P. K., Chakraborty, K., Roy, R., Jayaram, C., Sridevi, B., & Sarma, V. V. S. S. (2025). Long‐term changes of surface total alkalinity and its driving mechanisms in the north Indian Ocean. Global Biogeochemical Cycles, 39(8), e2024GB008344.

## Creators
### Prasanna Kanti Ghoshal (P. K. Ghoshal) and Dr Apurva Padamnabh Joshi (A. P. Joshi)

This repository contains the machine learning workflow used for the reconstruction of surface **Total Alkalinity (TA)** in the Indian Ocean using an ensemble of **XGBoost** models.

## Contents

* **Train-Testsplit.ipynb** – Data preprocessing and train-test splitting.
* **XGB_tuning.ipynb** – Hyperparameter optimization using Optuna.
* **XGB_training_new.ipynb** – Training of the XGBoost model.
* **XGB_shap_analysis.ipynb** – SHAP-based model interpretation and feature importance analysis.
* **Xgb_pred_BoB_final.py** – Gridded prediction and uncertainty generation.
* **TA_Paper_Figs.ipynb** & **TA_Paper_Figs1.ipynb** – Figure generation for manuscript preparation.

## Predictors

* Sea Surface Temperature (SST)
* Sea Surface Salinity (SSS)
* Mixed Layer Depth (MLD)

## Methods

* XGBoost Regression
* Ensemble Learning
* Cross-Validation
* Monte Carlo Uncertainty Analysis
* SHAP Explainability

## Requirements

```bash
numpy
pandas
xarray
matplotlib
scikit-learn
xgboost
optuna
joblib
shap
scipy
```

