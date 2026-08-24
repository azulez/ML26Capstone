# /// script
# dependencies = [
#     "transformers",
#     "torch",
# ]
# ///
"""Quick FinBERT sentiment check for a single sentence.

Usage:
    uv run finbert_sentiment.py "Company X beat earnings expectations."
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import pipeline


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} \"<sentence>\"")
        sys.exit(1)

    sentence = sys.argv[1]

    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device, top_k=None)
    scores = sorted(classifier(sentence)[0], key=lambda r: r["score"], reverse=True)

    print(f"Sentence: {sentence}")
    print(f"Device: {'CUDA' if device == 0 else 'CPU'}")
    print(f"Sentiment: {scores[0]['label']} (score: {scores[0]['score']:.4f})")
    for r in scores:
        print(f"  {r['label']}: {r['score']:.4f}")


if __name__ == "__main__":
    main()
