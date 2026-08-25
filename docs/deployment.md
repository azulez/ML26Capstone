# Deployment: earnings-call excess-return inference API

## Architecture

```
Browser (deploy/frontend/index.html, hosted on S3 static website)
    |  fetch() with X-Api-Key header
    v
Always-on EC2 instance (c6i.large), plain HTTP on port 5000
    - server/app.py: Flask app, gunicorn (1 worker)
      parse_turns -> TurnEncoder -> AttentionPoolingClassifier -> sigmoid
    - Swagger UI at /apidocs/ (see "Why Flasgger" below)
```

`server/app.py` is the single source of truth for serving: it runs
unmodified locally (`uv run python server/app.py`) and on EC2, importing
`src/modeling/*` the same way `notebooks/` and `data/generated/` producing
scripts already do (no packaging, just `sys.path` -- this project has no
`[build-system]` in `pyproject.toml` and isn't meant to be installed).

### Why EC2, not Lambda + API Gateway (history)

This stack originally ran as a container-image Lambda behind API Gateway.
That was abandoned after a real fix-forward investigation showed it was
fundamentally unfixable: real transcripts take 30-60s of actual FinBERT
inference (batched forward passes over dozens of speaker turns), which
exceeds API Gateway REST APIs' hard, non-configurable 29-second integration
timeout -- every non-trivial request would 504, warm or cold, regardless of
retries. Lambda's memory (hence CPU) is separately hard-capped at 3008MB
(~1.7 vCPU) on this AWS account, so more compute wasn't an option within
that architecture either. An always-on EC2 instance removes both problems:
the model loads once and stays resident (no cold start), and there's no
API-Gateway-style external timeout in the path at all.

### Why SageMaker wasn't used instead

SageMaker real-time endpoints have the same shape of problem, just with a
higher (still hard, still non-configurable) 60-second invoke timeout
instead of API Gateway's 29s -- not unlimited. They're also always-on
billed (same cost profile as EC2) but cost more per hour than equivalent
raw EC2 for the same instance family, and require repackaging the model
into SageMaker's serving container format. For one small model with no
need for SageMaker's versioning/A-B-testing/autoscaling features, that
extra cost and setup isn't worth it over plain EC2.

### Why Flasgger for Swagger docs

`server/app.py` has one real endpoint (`/predict`) with a simple
dict-in/dict-out JSON contract. **Flasgger** was chosen over the
alternatives because it fits that shape with the least ceremony: docs are
written as a YAML docstring directly on the route function, and it serves
an interactive Swagger UI at `/apidocs/` with no other code changes.
`flask-smorest` would add real value (request/response validation) at the
cost of Marshmallow schema classes -- overhead this project's single simple
endpoint doesn't need. `connexion` is spec-first (the OpenAPI YAML drives
routing) and is meant for APIs designed that way from the start, not a
one-endpoint retrofit.

## Known limitations

- **Accuracy is ~49-50% on held-out test data** (see
  `notebooks/model_training_full_dataset.ipynb`, Section B) -- effectively
  no better than a coin flip. This is deployed as a demonstration of the
  end-to-end pipeline (transcript text in, prediction out, on real AWS
  infrastructure), not as a source of real trading signal. The frontend
  displays this caveat directly next to the result.
- **No TLS.** The frontend fetches `http://<instance-ip>:5000/predict` in
  plain HTTP -- consistent with the S3 website frontend, which is also
  plain HTTP, so there's no mixed-content issue, but traffic (including the
  API key header) is unencrypted. Acceptable for a demo; would need an ALB
  + ACM certificate (and a domain) to fix properly.
- **Single gunicorn worker.** A single real inference run peaks around
  2.5GB RSS; `c6i.large` has 4GiB total, so running multiple workers (each
  loading its own FinBERT copy) risks OOM. This means one request is
  processed at a time -- concurrent requests queue. Fine for a single-user
  demo; mirrors the old Lambda's `ReservedConcurrentExecutions` intent to
  bound concurrency for cost/stability.
- **Checkpoint provenance**: the deployed checkpoint is
  `attention_pooling_model_full.pt`, produced by re-running Section B of
  `model_training_full_dataset.ipynb` (added as a save-checkpoint cell right
  after the `test_accuracy` cell) -- distinct from
  `attention_pooling_model.pt`, which came from the earlier sampled-subset
  finetuning notebook and reports different (and less trustworthy) numbers.
- **No real authentication.** The API key is embedded in the frontend's
  page source and visible to anyone who opens dev tools -- it deters casual
  bot scanning, nothing more. There's no request-rate throttle at the app
  layer (unlike the old API Gateway usage plan); cost containment is now
  the flat EC2 bill itself plus the optional CloudWatch billing alarm (see
  "Estimated cost" below) rather than a per-request quota.

## Prerequisites

Already covered in detail during account setup -- see the "Prerequisites
checklist" section of the design plan referenced in project history. In
short: an AWS account with billing alerts/budget configured, an IAM
identity to deploy with (not root), and credentials configured locally via
`aws configure`. `aws` CLI and `uv` must be on `PATH`. No `docker` or `sam`
needed -- this stack is plain CloudFormation plus an SSM-based code sync,
no container build.

## Running locally

`server/app.py` runs the same code path used on EC2:

```bash
uv run python server/app.py
# or: uv run --with-requirements server/requirements.txt gunicorn -w 1 --timeout 120 \
#       --chdir server -b 0.0.0.0:5000 app:app
```

By default `FINBERT_DIR` is unset, so `transformers` downloads and caches
`ProsusAI/finbert` (~440MB, one-time) from Hugging Face on first request.
Set `FINBERT_DIR` to a local path to skip the download if you already have
the weights (e.g. from an earlier deploy). `CHECKPOINT_PATH` defaults to
`data/generated/models/attention_pooling_model_full.pt` (see the runbook
below for how to produce it). No `API_KEY` env var means the `/predict`
route skips the key check entirely -- convenient for local dev.

Once running: `http://localhost:5000/health` for a liveness check,
`http://localhost:5000/apidocs/` for the Swagger UI (use "Try it out" to
POST a transcript directly from the browser -- this doubles as the
pre-deploy smoke test that `deploy/local_test.py` used to serve).

## Runbook

```bash
# One-time: produce the checkpoint the deploy script expects.
# Run the save-checkpoint cell in notebooks/model_training_full_dataset.ipynb
# (Section B, right after the test_accuracy cell) so
# data/generated/models/attention_pooling_model_full.pt exists.

# Deploy (idempotent -- safe to re-run after code changes; infra changes
# go through CloudFormation, app code is re-synced and the service
# restarted every time via SSM Run Command).
./deploy/deploy.sh

# ...or with a non-default instance size / the billing alarm enabled:
./deploy/deploy.sh --instance-type c6i.xlarge
./deploy/deploy.sh --enable-billing-alarm --alarm-email you@example.com
```

`deploy.sh` fails fast with a clear message if `aws`/`uv` aren't installed,
AWS credentials aren't configured, or the full-dataset checkpoint is
missing. On success it prints the frontend URL, the API invoke URL, and the
Swagger UI URL.

### Teardown

```bash
./deploy/teardown.sh
```

Deletes the CloudFormation stack (EC2 instance, security group, IAM role,
Elastic IP), the code S3 bucket, the API key SSM parameter, and the
frontend S3 bucket. **Run this once the assignment is graded** -- unlike
the old Lambda backend (pay-per-request, scales to zero), this instance
bills 24/7 until torn down. If the billing alarm was enabled, its SNS
topic/subscription and the CloudWatch alarm are part of the stack and are
removed with it.

## Required IAM permissions

Scoped to what `deploy.sh`/`teardown.sh` touch:

- **CloudFormation** -- create/update/delete stack, describe stacks
- **EC2** -- create/terminate instances, security groups, Elastic IPs;
  describe instances
- **IAM** -- create/delete the EC2 instance role + instance profile:
  `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:PutRolePolicy`,
  `iam:PassRole`, `iam:GetRole`, `iam:DeleteRole`,
  `iam:CreateInstanceProfile`, `iam:DeleteInstanceProfile`
- **Systems Manager** -- `ssm:SendCommand`, `ssm:GetCommandInvocation`,
  `ssm:DescribeInstanceInformation` (deploy-code sync), `ssm:PutParameter`,
  `ssm:GetParameter`, `ssm:DeleteParameter` (API key storage)
- **S3** -- one bucket for the app-code tarball and one for the static
  frontend (public read via bucket policy + public access block disabled,
  static website hosting enabled)
- **CloudWatch** -- if the billing alarm is enabled
- **SNS** -- topic + email subscription for the billing alarm, if enabled

The EC2 instance's own role (attached automatically, not something you need
separately) only needs the `AmazonSSMManagedInstanceCore` managed policy
(so SSM Run Command can reach it) plus `s3:GetObject` scoped to the code
bucket.

## Billing alarm caveat

`AWS/Billing EstimatedCharges` only publishes in `us-east-1`, regardless of
which region the rest of the stack deploys to. If you deploy this stack
outside `us-east-1` with `--enable-billing-alarm`, the alarm resource will
be created but the underlying metric won't populate there -- deploy the
whole stack in `us-east-1`, or rely on the AWS Budget alert set up
separately during account setup.

## Estimated cost

Unlike the old Lambda backend, this is a **flat, always-on cost** -- there's
no scale-to-zero, and no per-request quota to bound worst-case usage,
because the instance bills the same whether it serves 0 or 1,000 requests:

- **EC2 `c6i.large`** (2 vCPU, 4GiB), on-demand, `us-east-1`: roughly
  $0.085/hr x 730 hr/month ~= **$62/month** if left running continuously.
  This is the dominant cost by far.
- **EBS (20GB gp3 root volume)**: ~$1.60/month.
- **Elastic IP**: free while associated with a running instance.
- **S3 (code + frontend buckets), SSM, CloudWatch Logs**: negligible
  (low single-digit KB/MB of traffic).

Total: **roughly $60-65/month** while the instance runs. The billing alarm
threshold defaults to $75 to sit just above that expected baseline (a
tripwire for misconfiguration, e.g. an oversized instance type, not the
expected steady state). **Run `./deploy/teardown.sh` when not actively
using/demoing this** -- there's no cheaper "idle" state short of full
teardown, since the whole point of this architecture is staying resident
to avoid cold starts. If you want to pause spend without a full teardown
(e.g. between demo sessions, keeping the Elastic IP/config), you can
manually stop the instance via the EC2 console -- but a stopped instance
loses the resident/warm state, so restarting reintroduces a wait (just not
a hard external timeout, since there's no gateway forcing one).
