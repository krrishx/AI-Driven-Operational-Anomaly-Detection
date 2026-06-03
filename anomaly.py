import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA

df = pd.read_excel(
    "master_exhauster_dataset.xlsx"
)

# FEATURES & TARGET

features = [
    "HT_Motor_Current",
    "Air_Flow",
    "Outlet_Temp",
    "Vibration_101",
    "Water_Flow",
    "FB2_Temp"
]
target = "FB1_Temp"

# REMOVE NULL VALUES

df = df.dropna(
    subset=features + [target]
)

# INPUTS & OUTPUT

X = df[features]
y = df[target]

# TRAIN REGRESSION MODEL

model = LinearRegression()
model.fit(X, y)

# PREDICTIONS

df["Predicted_FB1"] = model.predict(X)

# RESIDUAL CALCULATION

df["Residual"] = (
    df["FB1_Temp"] -
    df["Predicted_FB1"]
)

# DYNAMIC THRESHOLD

residual_mean = df["Residual"].mean()
residual_std = df["Residual"].std()
threshold = (
    residual_mean +
    (2 * residual_std)
)

# ANOMALY DETECTION

df["Anomaly"] = np.where(
    abs(df["Residual"]) > threshold,
    1,
    0
)

# HEALTH SCORE

df["Health_Score"] = (
    100 -
    (
        abs(df["Residual"]) /
        (threshold * 2)
    ) * 100
)
df["Health_Score"] = df[
    "Health_Score"
].clip(
    lower=0,
    upper=100
)

# MODEL METRICS

mae = mean_absolute_error(
    y,
    df["Predicted_FB1"]
)
r2 = r2_score(
    y,
    df["Predicted_FB1"]
)

# PRINT RESULTS

print("\n===================================")
print("SOFT SENSOR MODEL TRAINED")
print("===================================")
print(f"\nMAE: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"\nResidual Threshold: {threshold:.2f}")
print("\nANOMALY COUNT:")
print(df["Anomaly"].value_counts())

# SHOW ANOMALY ROWS

print("\n===================================")
print("ANOMALY ROWS")
print("===================================")
anomalies = df[df["Anomaly"] == 1]
print(
    anomalies[
        [
            "FB1_Temp",
            "Predicted_FB1",
            "Residual",
            "Health_Score"
        ]
    ]
)

# SAVE OUTPUT DATA

df.to_excel(
    "processed_exhauster_output.xlsx",
    index=False
)
print("\nProcessed file saved successfully.")

# VISUALIZATION SETTINGS

sns.set_style("darkgrid")

# ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(14,6))
plt.plot(
    df["FB1_Temp"].values,
    label="Actual FB1 Temp",
    linewidth=2
)
plt.plot(
    df["Predicted_FB1"].values,
    label="Predicted FB1 Temp",
    linewidth=2
)
plt.title(
    "Actual vs Predicted Bearing Temperature",
    fontsize=16
)
plt.xlabel("Data Points")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()

# RESIDUAL TREND GRAPH

plt.figure(figsize=(14,6))
plt.plot(
    df["Residual"].values,
    color="purple",
    linewidth=2,
    label="Residual"
)

# THRESHOLD LINES

plt.axhline(
    y=threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Upper Threshold"
)
plt.axhline(
    y=-threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Lower Threshold"
)

# ANOMALY POINTS

anomaly_points = df[
    df["Anomaly"] == 1
]
plt.scatter(
    anomaly_points.index,
    anomaly_points["Residual"],
    color="red",
    s=100,
    label="Anomalies"
)
plt.title(
    "Residual Trend Analysis",
    fontsize=16
)
plt.xlabel("Data Points")
plt.ylabel("Residual Error")
plt.legend()
plt.tight_layout()
plt.show()

# ERROR DISTRIBUTION

plt.figure(figsize=(10,5))
sns.histplot(
    df["Residual"],
    bins=25,
    kde=True
)
plt.title(
    "Residual Error Distribution",
    fontsize=16
)
plt.xlabel("Residual Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# CORRELATION HEATMAP

plt.figure(figsize=(10,7))
correlation_matrix = df[
    features + [target]
].corr()
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title(
    "Feature Correlation Heatmap",
    fontsize=16
)
plt.tight_layout()
plt.show()