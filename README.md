# Offshore Wind Turbine Predictive Maintenance (30-Day Early Warning)

An end-to-end machine learning and interactive dashboard framework for **SCADA anomaly detection** and **30-day main bearing failure forecasting** across wind turbine fleets, built using the **Fraunhofer CARE to Compare** real-world SCADA dataset.

## Problem Statement
Unscheduled main bearing failures cause costly offshore downtime and crane deployments. Fixed SCADA thresholds often trigger too late because sensor readings shift under changing wind and environmental conditions. This project models normal turbine behavior using turbine-specific z-scores and rolling trends to detect degradation up to **30 days before failure**.

## Approach
1. **Feature Engineering (`train.py`)**:
   - Computes turbine-specific z-scores from each turbine's healthy baseline (`train` split), plus **24-hour rolling means** and **24-hour trends** at hourly resolution.
   - Uses **Wind Farm B** (15 events, including 3 confirmed Main/Rotor Bearing failures and Rotor Bearing Temperature sensors `sensor_51`, `sensor_52`) and **Wind Farm A** (22 events, multi-component 30-day benchmark).
2. **Models & Leak-Free Validation**:
   - **Isolation Forest** (unsupervised anomaly detection trained only on healthy operation)
   - **Random Forest Classifier** (supervised 30-day failure risk prediction)
   - **Logistic Regression** (baseline comparison)
   - Evaluated via **5-fold Stratified Group Cross-Validation** grouped by `event_id` to prevent data leakage.
3. **Interactive Dashboard (`app.py`)**:
   - **250-Turbine Offshore Fleet Monitor** (fleet-level risk simulation grounded in validated Farm B outputs)
   - **Main Bearing Early Warning Timeline** & rotor bearing temperature curves
   - **False-Alarm vs. Recall Tradeoff** interactive threshold tuner

## Key Results
- **Wind Farm B — Main/Rotor Bearing (30-day horizon):** Random Forest achieves **ROC-AUC = 0.711**
- **Wind Farm A — Gearbox (30-day horizon):** Random Forest achieves **ROC-AUC = 0.929**

## Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py