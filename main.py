import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("master_exhauster_dataset.xlsx")

# Plot graph
plt.plot(df["Vibration_101"])

# Titles
plt.title("Vibration 101 Trend")
plt.xlabel("Rows")
plt.ylabel("Vibration")

# Show graph
plt.show()