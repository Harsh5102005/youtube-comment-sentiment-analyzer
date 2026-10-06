import os
import json, joblib, pandas as pd
import mlflow, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

pipe = joblib.load("models/model.pkl")
test = pd.read_csv("data/processed/test.csv")
pred = pipe.predict(test["clean"])

metrics = {
    "test_accuracy": accuracy_score(test["y"], pred),
    "test_macro_f1": f1_score(test["y"], pred, average="macro"),
}
Path("reports").mkdir(exist_ok=True)
json.dump(metrics, open("reports/metrics.json", "w"), indent=2)

fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay.from_predictions(test["y"], pred, display_labels=["Neg", "Neu", "Pos"], ax=ax)
fig.savefig("reports/confusion_matrix.png", bbox_inches="tight")

# attach the test results to the same MLflow run that trained the model
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
with mlflow.start_run(run_id=Path("models/run_id.txt").read_text().strip()):
    mlflow.log_metrics(metrics)
    mlflow.log_figure(fig, "confusion_matrix.png")
print(metrics)