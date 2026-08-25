"""Flask serving app: raw earnings-call transcript text -> excess-return direction.

Wraps the same speaker-turn attention-pooling pipeline used for training
(`src/modeling/turns.py`, `encoding.py`, `dataset.py`, `model.py`) around a
single-transcript inference path. Runs identically locally and on the
deployed EC2 instance (see `docs/deployment.md`) -- FinBERT is downloaded
and cached by `transformers` on first run rather than baked into an image,
so there's no separate build step for either environment.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from flasgger import Swagger
from flask import Flask, jsonify, request

from modeling.dataset import PAD_ROLE_ID, ROLE_VOCAB
from modeling.encoding import TurnEncoder
from modeling.model import AttentionPoolingClassifier
from modeling.turns import parse_turns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FINBERT_DIR = os.environ.get("FINBERT_DIR", "ProsusAI/finbert")
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", str(PROJECT_ROOT / "data" / "generated" / "models" / "attention_pooling_model_full.pt")
)
API_KEY = os.environ.get("API_KEY")

app = Flask(__name__)
app.config["SWAGGER"] = {
    "title": "Earnings-Call Excess-Return API",
    "uiversion": 3,
    "specs_route": "/apidocs/",
}
Swagger(app)

_encoder: TurnEncoder | None = None
_model: AttentionPoolingClassifier | None = None


def _load_model() -> tuple[TurnEncoder, AttentionPoolingClassifier]:
    global _encoder, _model
    if _model is None:
        logger.info("Loading FinBERT from %s", FINBERT_DIR)
        _encoder = TurnEncoder(model_name=FINBERT_DIR, device="cpu")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model = AttentionPoolingClassifier(
            embedding_dim=checkpoint["embedding_dim"],
            role_dim=checkpoint["role_dim"],
            attention_dim=checkpoint["attention_dim"],
            dropout=checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _model = model
        logger.info("Model load complete")
    return _encoder, _model


def predict(transcript_text: str) -> dict:
    turns = parse_turns(transcript_text)
    if not turns:
        raise ValueError("No speaker turns found in transcript text")

    encoder, model = _load_model()
    encoded = encoder.encode([turn.text for turn in turns])

    role_ids = [ROLE_VOCAB.get(turn.role.value, PAD_ROLE_ID) for turn in turns]

    embeddings = torch.tensor(encoded.embeddings, dtype=torch.float32).unsqueeze(0)  # [1, T, H]
    role_id_tensor = torch.tensor(role_ids, dtype=torch.long).unsqueeze(0)  # [1, T]
    mask = torch.ones(1, len(turns), dtype=torch.bool)  # [1, T]

    with torch.no_grad():
        logits, attention_weights = model(embeddings, role_id_tensor, mask)
        probability = torch.sigmoid(logits).item()

    return {
        "probability": probability,
        "predicted_label": int(probability > 0.5),
        "num_turns": len(turns),
        "attention_weights": attention_weights.squeeze(0).tolist(),
    }


@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,X-Api-Key"
    response.headers["Access-Control-Allow-Methods"] = "POST,OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health():
    """Liveness check.
    ---
    responses:
      200:
        description: Service is up.
    """
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_route():
    """Score an earnings-call transcript for predicted excess-return direction.
    ---
    parameters:
      - name: X-Api-Key
        in: header
        type: string
        required: true
        description: Shared API key (deters casual bot scanning, not real auth).
      - name: body
        in: body
        required: true
        schema:
          type: object
          required: [transcript]
          properties:
            transcript:
              type: string
              description: Raw earnings-call transcript text.
    responses:
      200:
        description: Prediction result.
        schema:
          type: object
          properties:
            probability:
              type: number
            predicted_label:
              type: integer
            num_turns:
              type: integer
            attention_weights:
              type: array
              items:
                type: number
      400:
        description: Missing/invalid request body, or no speaker turns found.
      401:
        description: Missing or incorrect X-Api-Key header.
      500:
        description: Internal error while scoring the transcript.
    """
    if request.method == "OPTIONS":
        return "", 200

    if API_KEY and request.headers.get("X-Api-Key") != API_KEY:
        return jsonify({"error": "Missing or invalid X-Api-Key header"}), 401

    body = request.get_json(silent=True) or {}
    transcript_text = (body.get("transcript") or "").strip()
    if not transcript_text:
        return jsonify({"error": "Missing or empty 'transcript' field"}), 400

    try:
        result = predict(transcript_text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Unhandled error during prediction")
        return jsonify({"error": "Internal error while scoring transcript"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
