# AI Driven Operational Anomaly Detection

An industrial machine health monitoring and anomaly detection system developed during my internship at **SAIL Bokaro Steel Plant (C&IT Department)**.

The project uses Machine Learning and operational process data from Sinter Plant Exhausters to identify abnormal operating conditions, assess machine health, and analyze shift-wise performance through an interactive dashboard.

---

## Problem Statement

Sinter Plant Exhausters are critical equipment in the steel manufacturing process. Unexpected deviations in operating parameters can lead to reduced efficiency, increased maintenance costs, and unplanned downtime.

The objective of this project was to develop a data-driven framework capable of:

* Detecting operational anomalies
* Monitoring machine health
* Tracking equipment performance
* Providing shift-wise efficiency insights
* Supporting predictive maintenance initiatives

---

## Key Features

* Machine Learning based temperature prediction
* Residual-based anomaly detection
* Machine Health Intelligence System
* Health score generation
* Multi-sensor vibration monitoring
* Bearing temperature analysis
* Pressure and oil health monitoring
* Shift-wise efficiency analysis
* Interactive Streamlit dashboard

---

## Machine Learning Workflow

1. Historical exhauster operational data was collected and preprocessed.
2. Relevant process parameters were selected as model inputs.
3. A Linear Regression model was trained to predict FB1 Bearing Temperature.
4. Residual values were calculated using actual and predicted temperatures.
5. Dynamic thresholds were applied to identify abnormal operating conditions.
6. Machine health scores were generated using residual-based analysis.
7. Results were visualized through an interactive dashboard.

---

## Technologies Used

| Category         | Technologies  |
| ---------------- | ------------- |
| Programming      | Python        |
| Data Processing  | Pandas, NumPy |
| Machine Learning | Scikit-Learn  |
| Visualization    | Plotly        |
| Dashboard        | Streamlit     |
| File Handling    | OpenPyXL      |

---

## Key Outcomes

* Accurate bearing temperature prediction
* Automated anomaly detection
* Machine health assessment
* Interactive operational dashboard
* Shift-wise performance evaluation
* Improved visibility into equipment behavior

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run dashboard.py
```

---

## Internship Details

**Organization:** SAIL Bokaro Steel Plant
**Department:** Computer & Information Technology (C&IT)
**Duration:** 11 May 2026 – 6 June 2026

---

