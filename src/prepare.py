import pandas as pd, yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from text_utils import preprocess

p = yaml.safe_load(open("params.yaml"))["data"]
LABELS = {"Negative": 0, "Neutral": 1, "Positive": 2}

df = pd.read_csv(p["raw_path"]).rename(columns={"CommentText": "text", "Sentiment": "label"})
df = df.dropna().drop_duplicates()

conflict = df.groupby("text")["label"].nunique()
df = df[~df["text"].isin(conflict[conflict > 1].index)]

df = df[df["text"].str.split().str.len().between(1, p["max_words"])]
df["y"] = df["label"].map(LABELS)
df["clean"] = df["text"].apply(preprocess)
df = df[df["clean"].str.len() > 0][["text", "clean", "y"]].reset_index(drop=True)

train_val, test = train_test_split(df, test_size=p["test_size"], stratify=df["y"], random_state=p["seed"])
train, val = train_test_split(train_val, test_size=p["val_size"], stratify=train_val["y"], random_state=p["seed"])

out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
for name, d in [("train", train), ("val", val), ("test", test)]:
    d.to_csv(out / f"{name}.csv", index=False)
    print(name, d.shape)