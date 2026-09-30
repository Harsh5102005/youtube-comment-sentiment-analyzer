import json, sys, yaml
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

P = yaml.safe_load(open("params.yaml"))["register"]
NAME = P["model_name"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

run_id = Path("models/run_id.txt").read_text().strip()
f1 = json.load(open("reports/metrics.json"))["test_macro_f1"]

# gate 1: minimum quality
if f1 < P["min_test_macro_f1"]:
    sys.exit(f"FAILED gate: test macro-F1 {f1:.4f} < {P['min_test_macro_f1']}")

# don't register the same run twice
if any(mv.run_id == run_id for mv in client.search_model_versions(f"name='{NAME}'")):
    sys.exit(0)

mv = mlflow.register_model(f"runs:/{run_id}/model", NAME)
client.set_model_version_tag(NAME, mv.version, "test_macro_f1", f"{f1:.4f}")

# gate 2: promote only if better than the current production model
try:
    prod = client.get_model_version_by_alias(NAME, "production")
    prod_f1 = client.get_run(prod.run_id).data.metrics.get("test_macro_f1", -1)
except Exception:
    prod_f1 = -1

if f1 > prod_f1:
    client.set_registered_model_alias(NAME, "production", mv.version)
    print(f"v{mv.version} promoted to production (macro-F1 {f1:.4f}, previous {prod_f1:.4f})")
else:
    print(f"v{mv.version} registered but NOT promoted ({f1:.4f} <= {prod_f1:.4f})")