# Model architecture: speaker-turn attention pooling

## Problem

`data/generated/labeled_transcripts.parquet` pairs each earnings-call transcript
with `excess_return_label` (stock return minus S&P 500 return over a lag
window). Transcripts average ~57k characters (~12-15k tokens), far past the
512-token window of a standard BERT-family encoder, so the document can't be
fed to the model whole. The chosen document itself already has a resolvable
structure that both keeps chunking meaningful and gives the model potential
signal that generic sliding-window chunking would erase.

## Why speaker turns, not fixed-size chunks

Transcripts are already segmented into speaker turns delimited by
`Name -- Title` header lines (see `_extract_raw_string`/parsing exploration
in `src/datawrangling/dateparsing.py` for the precedent of parsing this raw
Motley Fool format). Each turn is a natural semantic unit, and the speaker's
role (executive vs. IR vs. analyst) is recoverable from the title string.

This matters for two reasons:

1. **Long-document handling.** Encoding per turn instead of per fixed-size
   window avoids truncating mid-thought and keeps each unit small enough for
   a standard encoder's context window.
2. **Signal location.** Prepared remarks (scripted, IR-vetted) and Q&A
   responses (unscripted) carry different amounts of market-relevant signal.
   Finance-NLP literature associates executive tone/hedging under analyst
   questioning specifically with forward returns. Fixed-size chunking would
   mix these together indiscriminately; turn-level + role-tagged chunking
   keeps them separable.

## Architecture

1. **Turn extraction**: split each transcript on `Name -- Title` header
   lines; tag each turn with a coarse speaker role (CEO / CFO / other
   executive / IR / analyst), parsed from the title string.
2. **Turn encoder**: a FinBERT-family encoder produces one embedding per
   turn (frozen or fine-tuned — see the training notebook for which).
3. **Attention pooling**: a learned attention layer combines turn embeddings
   (conditioned on speaker-role) into a single document embedding. The
   attention weights double as an interpretable "which speaker/turn drove
   this prediction" signal — this is the "officer score" concept from the
   original design discussion, discovered by the model rather than
   hand-specified.
4. **Prediction head**: trained end-to-end against `excess_return_label`.

## Supervision caveat

The only ground-truth label is document-level (`excess_return_label`); there
is no per-turn label. The attention-pooling layer is therefore the only part
of the model that can legitimately be trained to discriminate between turns —
a per-turn classifier/regressor trained directly against the document label
would be a mislabeled multiple-instance-learning setup.

## Target framing

Primary target is classification (direction / confidence bucket) rather than
raw regression on `excess_return_label`, given the ~3k-row dataset size —
more stable to fit and evaluate. Magnitude (regression) is a secondary/
stretch head, matching the eventual product goal of "confidence it goes up
or down, and by how much."
