# ML2026: Earnings-Call Excess-Return Model Deployment Documentation

Predicts the direction of a stock's excess return following an earnings
call, from the raw call transcript text. A FinBERT-based
attention-pooling classifier (`src/modeling/`) is trained in
`notebooks/`, served by a Flask API (`server/app.py`), and deployed to
AWS via `deploy/`. See `docs/deployment.md` for full deployment details
(architecture, cost, IAM, teardown).

**Known limitation:** ~49-50% accuracy on held-out test data --
effectively a coin flip. This project demonstrates an end-to-end
pipeline (transcript in, prediction out) on real infrastructure, not a
source of real trading signal.

For additional project commentary and future potential plans, please
see the api_quicktest.ipynb notebook.

## Running a local instance

No AWS account needed for local use. From the repo root:

```bash
uv run python server/app.py
```

This starts the same Flask app that runs on EC2 in production, on
`http://localhost:5000`. Notes:

- **First run downloads FinBERT** (~440MB from Hugging Face) unless
  `FINBERT_DIR` points at a local copy.
- **`CHECKPOINT_PATH`** defaults to
  `data/generated/models/attention_pooling_model_full.pt`. Produce it by
  running the save-checkpoint cell in
  `notebooks/model_training_full_dataset.ipynb` (Section B, right after
  the `test_accuracy` cell) if it doesn't already exist.
- **No `API_KEY` env var** means `/predict` skips auth entirely --
  convenient for local dev.

## Using the site

Once the server is running:

- `http://localhost:5000/health` -- liveness check.
- `http://localhost:5000/apidocs/` -- interactive Swagger UI; use "Try
  it out" to POST a transcript and see a prediction directly from the
  browser.
- `POST http://localhost:5000/predict` with a JSON body
  `{"transcript": "<speaker turns as text>"}` returns the predicted
  probability, label, turn count, and per-turn attention weights.

The deployed version adds a static frontend (`deploy/frontend/`) and an
`X-Api-Key` header requirement -- see `docs/deployment.md` for deploying
to AWS with `deploy/deploy.sh`.

## Deploying to AWS

### Environment

Production runs the identical `server/app.py` Flask app (gunicorn, 1
worker) on a single always-on **EC2 `c6i.large`** instance, plain HTTP on
port 5000, with a static frontend (`deploy/frontend/`) hosted on an S3
website bucket in front of it:

```
Browser (S3 static website frontend)
    |  fetch() with X-Api-Key header
    v
EC2 instance -- server/app.py (Flask + gunicorn)
    -> Swagger UI at /apidocs/
```

EC2 was chosen over Lambda+API Gateway or SageMaker specifically because
real inference runs (30-60s of FinBERT forward passes) exceed both of
those services' hard, non-configurable invoke timeouts (29s and 60s
respectively) -- an always-on instance has no such external timeout. See
`docs/deployment.md` for the full rationale, cost breakdown, and known
limitations (no TLS, single worker, no real auth).

### The deploy tool

`deploy/deploy.sh` is a single idempotent script that provisions and
updates the whole stack:

1. Packages `server/`, `src/modeling/`, and the model checkpoint into a
   tarball and uploads it to an S3 code bucket.
2. Generates (or reuses) an API key stored in SSM Parameter Store.
3. Applies `deploy/template.yaml` via CloudFormation (EC2 instance,
   security group, IAM role, Elastic IP -- created or updated as needed).
4. Pushes the code tarball to the instance and restarts the service via
   SSM Run Command (no SSH keys, no open port 22).
5. Builds and publishes the static frontend to an S3 website bucket,
   injecting the live API URL/key.

```bash
./deploy/deploy.sh                                          # default c6i.large, no billing alarm
./deploy/deploy.sh --instance-type c6i.xlarge
./deploy/deploy.sh --enable-billing-alarm --alarm-email you@example.com
```

Re-running it is safe: infra changes go through CloudFormation, and app
code/service are always re-synced. `./deploy/teardown.sh` deletes the
entire stack (instance, buckets, API key param) -- run it once you're
done, since the instance bills 24/7 (~$60-65/month) until torn down.

### Prerequisites

- An AWS account with billing alerts/budget configured, and an IAM
  identity to deploy with (not root).
- AWS credentials configured locally (`aws configure`), including a
  default region.
- `aws` CLI and `uv` on `PATH`. No `docker` or `sam` needed.
- The model checkpoint at
  `data/generated/models/attention_pooling_model_full.pt` -- produced by
  the save-checkpoint cell in
  `notebooks/model_training_full_dataset.ipynb` (Section B). The deploy
  script fails fast with a clear error if it's missing.
- IAM permissions covering CloudFormation, EC2, IAM (role/instance
  profile creation), SSM, S3, and (if using the billing alarm)
  CloudWatch/SNS -- see the "Required IAM permissions" section of
  `docs/deployment.md` for the exact scoped list.

## Exercising the API from a notebook

`notebooks/api_quicktest.ipynb` is the fastest way to hit either a local
or deployed API without AWS credentials: set `API_URL` (and `API_KEY` if
required) by hand at the top, then run the notebook to send a sample
transcript and print the response. Point `API_URL` at
`http://localhost:5000/predict` for a local instance, or at a deployed
instance's address.
