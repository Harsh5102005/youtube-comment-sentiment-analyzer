import os
import joblib, yaml, pandas as pd
import mlflow, mlflow.sklearn
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score

P = yaml.safe_load(open("params.yaml"))["baseline"]

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

features = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=tuple(P["word_ngram"]), min_df=2, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(P["char_ngram"]), min_df=3, sublinear_tf=True)),
])
pipe = Pipeline([("features", features), ("clf", LogisticRegression(max_iter=1000, C=P["C"]))])

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("yt-sentiment")

with mlflow.start_run(run_name="baseline-lr-word-char") as run:
    mlflow.log_params({"model": "tfidf_word_char+logreg", **P})
    mlflow.log_param("n_train", len(train))

    pipe.fit(train["clean"], train["y"])
    pred = pipe.predict(val["clean"])
    mlflow.log_metrics({
        "val_accuracy": accuracy_score(val["y"], pred),
        "val_macro_f1": f1_score(val["y"], pred, average="macro"),
    })
    mlflow.sklearn.log_model(pipe, name="model")     # on older MLflow use artifact_path="model"

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/model.pkl")
    Path("models/run_id.txt").write_text(run.info.run_id)
    print("run:", run.info.run_id, "| val macro-F1:", f1_score(val["y"], pred, average="macro"))