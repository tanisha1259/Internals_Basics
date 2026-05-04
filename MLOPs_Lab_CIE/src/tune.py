import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("../data/training_data.csv")

X = df.drop("mix_quality_score", axis=1)
y = df["mix_quality_score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = GradientBoostingRegressor()

# Parameter grid
param_dist = {
    "n_estimators": [50, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

mlflow.set_experiment("soundforge-mix-quality-score")

with mlflow.start_run(run_name="tuning-soundforge") as parent_run:

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=5,   # keep small for speed
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    best_mae = mean_absolute_error(y_test, preds)

    # Log best params
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_mae", best_mae)

    # Save JSON
    import os, json
    os.makedirs("../results", exist_ok=True)

    output = {
        "search_type": "random",
        "n_folds": 3,
        "total_trials": 5,
        "best_params": search.best_params_,
        "best_mae": best_mae,
        "best_cv_mae": -search.best_score_,
        "parent_run_name": "tuning-soundforge"
    }

    with open("../results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("Task 2 completed")