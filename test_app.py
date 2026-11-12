import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app import app

SAMPLE = load_breast_cancer().data[0].tolist()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan -> loads model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert len(r.json()["feature_names"]) == 30


def test_predict_schema_and_200(client):
    r = client.post("/predict", json={"features": SAMPLE})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "label", "probabilities"}
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("malignant", "benign")
    assert pytest.approx(sum(body["probabilities"].values()), abs=1e-6) == 1.0
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"].values())


def test_predict_known_sample_is_malignant(client):
    # First row of the dataset is a malignant case
    r = client.post("/predict", json={"features": SAMPLE})
    assert r.json()["label"] == "malignant"


def test_wrong_feature_count_returns_422(client):
    r = client.post("/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 422


def test_missing_field_returns_422(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_non_numeric_returns_422(client):
    r = client.post("/predict", json={"features": ["a"] * 30})
    assert r.status_code == 422


def test_wrong_method_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_unknown_route_returns_404(client):
    assert client.get("/nope").status_code == 404
