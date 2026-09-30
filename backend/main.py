import os, sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from text_utils import preprocess

# default: the mlflow.db in the project root, whatever directory you start from
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MODEL_URI = os.getenv("MODEL_URI", "models:/yt-sentiment@production")
LABELS = ["Negative", "Neutral", "Positive"]
MAX_COMMENTS = 500

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)   # loaded once at startup
    yield
    state.clear()

app = FastAPI(title="YT Sentiment API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Comment(BaseModel):
    text: str
    timestamp: Optional[str] = None

class PredictRequest(BaseModel):
    comments: List[Comment] = Field(..., max_length=MAX_COMMENTS)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.comments:
        raise HTTPException(400, "No comments provided")
    clean = [preprocess(c.text) for c in req.comments]
    preds = state["model"].predict(clean)
    results = [{"text": c.text, "timestamp": c.timestamp, "sentiment": LABELS[int(p)]}
               for c, p in zip(req.comments, preds)]
    counts = {l: sum(r["sentiment"] == l for r in results) for l in LABELS}
    return {"results": results, "counts": counts, "total": len(results)}