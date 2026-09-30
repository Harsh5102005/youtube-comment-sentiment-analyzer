import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # the "with" block runs startup, so the model loads
        yield c

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

def test_predict_basic(client):
    r = client.post("/predict", json={"comments": [{"text": "I love this, so helpful!"},
                                                   {"text": "This is garbage"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert sum(body["counts"].values()) == 2
    assert all(x["sentiment"] in {"Negative", "Neutral", "Positive"} for x in body["results"])

def test_predict_empty_list(client):
    assert client.post("/predict", json={"comments": []}).status_code == 400

def test_predict_too_many(client):
    many = [{"text": "ok"}] * 501
    assert client.post("/predict", json={"comments": many}).status_code == 422

def test_emoji_and_weird_text(client):
    r = client.post("/predict", json={"comments": [{"text": "😂😂😂"}, {"text": "   "}, {"text": "http://x.com"}]})
    assert r.status_code == 200