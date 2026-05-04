import pandas as pd
import numpy as np
import os, json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load original data
df1 = pd.read_csv("../data/training_data.csv")

# Load new data
df2 = pd.read_csv("../data/new_data.csv")

# Combine
df = pd.concat([df1, df2], ignore_index=True)

X = df.drop("mix_quality_score", axis=1)
y = df["mix_quality_score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Retrain model (same as best model)
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

new_rmse = np.sqrt(mean_squared_error(y_test, preds))

# OLD RMSE from Task 1 (hardcode)
old_rmse = 7.369915966890071

improvement = old_rmse - new_rmse

# Decision
if improvement >= 0.3:
    decision = "promoted"
else:
    decision = "kept_champion"

# Save JSON
os.makedirs("../results", exist_ok=True)

output = {
    "original_data_rows": len(df1),
    "new_data_rows": len(df2),
    "combined_data_rows": len(df),
    "champion_rmse": old_rmse,
    "retrained_rmse": new_rmse,
    "improvement": improvement,
    "min_improvement_threshold": 0.3,
    "action": decision,
    "comparison_metric": "rmse"
}

with open("../results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed")