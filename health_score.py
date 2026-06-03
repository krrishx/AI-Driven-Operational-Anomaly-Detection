import pandas as pd

# Load dataset
df = pd.read_excel("master_exhauster_dataset.xlsx")

# Important averages
avg_vibration = df["Vibration_101"].mean()
avg_temp = df["Outlet_Temp"].mean()
avg_airflow = df["Air_Flow"].mean()

# Start with perfect health
health_score = 100

# Reduce score based on conditions

# High vibration
if avg_vibration > 1.2:
    health_score -= 20

# High temperature
if avg_temp > 140:
    health_score -= 20

# Low airflow
if avg_airflow < 19000:
    health_score -= 20

# Decide machine status
if health_score >= 80:
    status = "Healthy ✅"

elif health_score >= 60:
    status = "Warning ⚠️"

else:
    status = "Critical 🔴"

# Print results
print("Machine Health Score:", health_score)
print("Machine Status:", status)