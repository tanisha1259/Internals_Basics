import mlflow
from mlflow.tracking import MlflowClient
import os, json

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name("soundforge-mix-quality-score")
experiment_id = experiment.experiment_id

# 🔥 Get correct run (GradientBoosting ONLY)
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="attributes.run_name = 'GradientBoosting'",
    order_by=["attributes.start_time DESC"],
    max_results=1
)

run = runs[0]
run_id = run.info.run_id

# 🔥 Correct model path
model_uri = f"runs:/{run_id}/model"

# Register model
model_name = "soundforge-mix-quality-score-predictor"

result = mlflow.register_model(
    model_uri,
    model_name
)

version = result.version

# Save JSON
os.makedirs("../results", exist_ok=True)

output = {
    "registered_model_name": model_name,
    "version": version,
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": run.data.metrics.get("mae", 0)
}

with open("../results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 completed")