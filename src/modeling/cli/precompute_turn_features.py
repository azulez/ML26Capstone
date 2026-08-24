"""CLI: parse transcripts into speaker turns, encode each turn with FinBERT, and cache the
results to disk. This is the expensive step (one FinBERT forward pass per turn, ~75 turns
per transcript) that the training notebook is not meant to re-run on every edit; run this
once beforehand and the notebook only reads its output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from modeling.encoding import TurnEncoder
from modeling.turns import parse_turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-path", default="data/generated/labeled_transcripts.parquet")
    parser.add_argument("--output-dir", default="data/generated/turn_features")
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N transcripts (dev smoke test)")
    args = parser.parse_args()

    transcripts = pd.read_parquet(args.transcripts_path)
    if args.limit is not None:
        transcripts = transcripts.iloc[: args.limit]

    print(f"Parsing turns for {len(transcripts)} transcripts...")
    records = []
    turn_texts: list[str] = []
    for row_id, transcript in zip(transcripts.index, transcripts["transcript"]):
        for turn_idx, turn in enumerate(parse_turns(transcript)):
            records.append(
                {
                    "row_id": row_id,
                    "turn_idx": turn_idx,
                    "speaker": turn.speaker,
                    "title": turn.title,
                    "role": turn.role.value,
                    "section": turn.section,
                    "n_chars": len(turn.text),
                }
            )
            turn_texts.append(turn.text)

    if not records:
        raise RuntimeError("No turns parsed from any transcript; check --transcripts-path.")

    print(f"Parsed {len(records)} turns. Encoding with {args.model_name}...")
    encoder = TurnEncoder(model_name=args.model_name, batch_size=args.batch_size, max_length=args.max_length)
    encoded = encoder.encode(turn_texts, show_progress=True)

    metadata = pd.DataFrame.from_records(records)
    for label, col in zip(encoded.sentiment_labels, encoded.sentiment_probs.T):
        metadata[f"sentiment_{label}"] = col

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output_dir / "turn_metadata.parquet", engine="pyarrow", index=False)
    np.save(output_dir / "turn_embeddings.npy", encoded.embeddings)
    (output_dir / "sentiment_labels.txt").write_text("\n".join(encoded.sentiment_labels))

    print(f"Wrote {len(metadata)} turn records to {output_dir}/turn_metadata.parquet")
    print(f"Wrote embeddings {encoded.embeddings.shape} to {output_dir}/turn_embeddings.npy")


if __name__ == "__main__":
    main()
