"""Sanity tests for server/app.py, run through Flask's test client (no
network, no separate server process). Uses the real FinBERT encoder and
checkpoint, so these are slow the first time FinBERT isn't already cached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "server"))

import app as flask_app_module  # noqa: E402

SAMPLE_TRANSCRIPT = "Jane Doe -- Chief Executive Officer\nThis is a smoke test."


@pytest.fixture()
def client():
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_predict_missing_transcript_field(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 400


def test_predict_blank_transcript(client):
    resp = client.post("/predict", json={"transcript": "   "})
    assert resp.status_code == 400


def test_predict_no_speaker_turns(client):
    resp = client.post("/predict", json={"transcript": "no headers here, just plain prose."})
    assert resp.status_code == 400
    assert "No speaker turns" in resp.get_json()["error"]


def test_predict_success(client):
    resp = client.post("/predict", json={"transcript": SAMPLE_TRANSCRIPT})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["num_turns"] == 1
    assert body["predicted_label"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert len(body["attention_weights"]) == body["num_turns"]


def test_predict_options_preflight(client):
    resp = client.options("/predict")
    assert resp.status_code == 200


def test_cors_headers_present(client):
    resp = client.get("/health")
    assert resp.headers["Access-Control-Allow-Origin"] == "*"


def test_predict_no_api_key_required_when_unset(client, monkeypatch):
    monkeypatch.setattr(flask_app_module, "API_KEY", None)
    resp = client.post("/predict", json={"transcript": SAMPLE_TRANSCRIPT})
    assert resp.status_code == 200


def test_predict_rejects_missing_or_wrong_api_key(client, monkeypatch):
    monkeypatch.setattr(flask_app_module, "API_KEY", "secret-key")

    resp = client.post("/predict", json={"transcript": SAMPLE_TRANSCRIPT})
    assert resp.status_code == 401

    resp = client.post(
        "/predict",
        json={"transcript": SAMPLE_TRANSCRIPT},
        headers={"X-Api-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_predict_accepts_correct_api_key(client, monkeypatch):
    monkeypatch.setattr(flask_app_module, "API_KEY", "secret-key")
    resp = client.post(
        "/predict",
        json={"transcript": SAMPLE_TRANSCRIPT},
        headers={"X-Api-Key": "secret-key"},
    )
    assert resp.status_code == 200
