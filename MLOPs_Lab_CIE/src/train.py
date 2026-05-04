import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("../data/training_data.csv")

X = df.drop("mix_quality_score", axis=1)
y = df["mix_quality_score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("soundforge-mix-quality-score")

results = []

models = {
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log
        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("team", "ml_engineering")

        mlflow.sklearn.log_model(model, "model")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

# Select best model (LOWEST MAE)
best_model = min(results, key=lambda x: x["mae"])

output = {
    "experiment_name": "soundforge-mix-quality-score",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "mae",
    "best_metric_value": best_model["mae"]
}

# Save JSON
import os, json
os.makedirs("../results", exist_ok=True)

with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")