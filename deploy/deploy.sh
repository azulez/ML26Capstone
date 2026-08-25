#!/usr/bin/env bash
# Deploys the earnings-call inference stack: an always-on EC2 instance
# running server/app.py (Flask), plus the static frontend on S3.
#
# Usage:
#   ./deploy/deploy.sh                                          # deploy without a billing alarm
#   ./deploy/deploy.sh --enable-billing-alarm --alarm-email you@example.com
#   ./deploy/deploy.sh --instance-type c6i.xlarge
#
# Code (server/, src/modeling/, the checkpoint) is pushed to S3 and pulled
# onto the instance via SSM Run Command -- no SSH keys, no port 22 open.
# Safe to re-run: infra changes go through CloudFormation, app code is
# re-synced and the service restarted every time.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEPLOY_DIR/.." && pwd)"

STACK_NAME="ml26-fintc"
CODE_BUCKET="${CODE_BUCKET:-ml2026-earnings-call-code}"
FRONTEND_BUCKET="${FRONTEND_BUCKET:-ml2026-earnings-call-frontend}"
API_KEY_PARAM="/ml26-fintc/api-key"

INSTANCE_TYPE="c6i.large"
ENABLE_BILLING_ALARM="false"
ALARM_EMAIL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-billing-alarm) ENABLE_BILLING_ALARM="true"; shift ;;
    --alarm-email) ALARM_EMAIL="$2"; shift 2 ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ "$ENABLE_BILLING_ALARM" == "true" && -z "$ALARM_EMAIL" ]]; then
  echo "ERROR: --enable-billing-alarm requires --alarm-email <address>" >&2
  exit 1
fi

echo "== Preflight checks =="

for tool in aws uv; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: '$tool' is not on PATH." >&2
    exit 1
  fi
done

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "ERROR: AWS credentials are not configured. Run 'aws configure' first." >&2
  exit 1
fi

CHECKPOINT="$REPO_ROOT/data/generated/models/attention_pooling_model_full.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  cat >&2 <<EOF
ERROR: Missing checkpoint: $CHECKPOINT
Run the save-checkpoint cell in notebooks/model_training_full_dataset.ipynb
(Section B, right after the test_accuracy cell) to produce it, then re-run
this script.
EOF
  exit 1
fi

REGION=$(aws configure get region)
if [[ -z "$REGION" ]]; then
  echo "ERROR: no default AWS region configured (run 'aws configure')." >&2
  exit 1
fi

echo "All preflight checks passed."

echo "== Packaging app code =="
BUILD_DIR="$(mktemp -d)"
mkdir -p "$BUILD_DIR/app/data/generated/models"
cp -r "$REPO_ROOT/server" "$BUILD_DIR/app/server"
find "$BUILD_DIR/app/server" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$BUILD_DIR/app/src"
cp -r "$REPO_ROOT/src/modeling" "$BUILD_DIR/app/src/modeling"
find "$BUILD_DIR/app/src" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
cp "$CHECKPOINT" "$BUILD_DIR/app/data/generated/models/attention_pooling_model_full.pt"

tar -C "$BUILD_DIR/app" -czf "$BUILD_DIR/app.tar.gz" .

aws s3 mb "s3://$CODE_BUCKET" --region "$REGION" 2>/dev/null || true
aws s3 cp "$BUILD_DIR/app.tar.gz" "s3://$CODE_BUCKET/app.tar.gz"

echo "== API key =="
API_KEY_VALUE=$(aws ssm get-parameter --name "$API_KEY_PARAM" --with-decryption \
  --query "Parameter.Value" --output text 2>/dev/null || true)
if [[ -z "$API_KEY_VALUE" || "$API_KEY_VALUE" == "None" ]]; then
  echo "Generating a new API key..."
  API_KEY_VALUE=$(openssl rand -hex 32)
  aws ssm put-parameter --name "$API_KEY_PARAM" --type SecureString --value "$API_KEY_VALUE" --overwrite >/dev/null
else
  echo "Reusing existing API key from SSM Parameter Store."
fi

echo "== CloudFormation deploy =="
CFN_ARGS=(
  --stack-name "$STACK_NAME"
  --template-file "$DEPLOY_DIR/template.yaml"
  --capabilities CAPABILITY_IAM
  --parameter-overrides
  "InstanceType=$INSTANCE_TYPE"
  "CodeBucketName=$CODE_BUCKET"
  "EnableBillingAlarm=$ENABLE_BILLING_ALARM"
)
if [[ -n "$ALARM_EMAIL" ]]; then
  CFN_ARGS+=("AlarmEmail=$ALARM_EMAIL")
fi

deploy_output=$(aws cloudformation deploy "${CFN_ARGS[@]}" 2>&1) || {
  if grep -q "No changes to deploy" <<<"$deploy_output"; then
    echo "$deploy_output"
    echo "(no infra changes -- continuing to sync app code)"
  else
    echo "$deploy_output" >&2
    exit 1
  fi
}
echo "$deploy_output"

echo "== Fetching stack outputs =="
INSTANCE_ID=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)
PUBLIC_IP=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='PublicIp'].OutputValue" --output text)
API_URL=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ApiInvokeUrl'].OutputValue" --output text)
echo "Instance: $INSTANCE_ID ($PUBLIC_IP)"

echo "== Waiting for SSM agent to register (new instances take ~1-2 min) =="
for i in $(seq 1 30); do
  STATUS=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --query "InstanceInformationList[0].PingStatus" --output text 2>/dev/null || true)
  if [[ "$STATUS" == "Online" ]]; then
    break
  fi
  sleep 10
done
if [[ "$STATUS" != "Online" ]]; then
  echo "ERROR: instance never registered with SSM (waited 5 minutes). Check the instance's UserData ran cleanly (EC2 console -> instance -> System Log)." >&2
  exit 1
fi

echo "== Syncing app code + restarting service via SSM =="
REMOTE_CMD=$(cat <<REMOTE
set -euxo pipefail
aws s3 cp s3://$CODE_BUCKET/app.tar.gz /tmp/app.tar.gz
rm -rf /opt/app
mkdir -p /opt/app
tar xzf /tmp/app.tar.gz -C /opt/app
chown -R ec2-user:ec2-user /opt/app
export PATH="/usr/local/bin:\$PATH"
runuser -l ec2-user -c 'cd /opt/app && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r server/requirements.txt'
printf 'API_KEY=%s\n' "$API_KEY_VALUE" > /etc/ml26-fintc.env
chmod 600 /etc/ml26-fintc.env
cp /opt/app/server/ml26-fintc.service /etc/systemd/system/ml26-fintc.service
systemctl daemon-reload
systemctl enable ml26-fintc
systemctl restart ml26-fintc
REMOTE
)

SSM_PARAMS_FILE="$(mktemp)"
python3 -c "
import json, sys
print(json.dumps({'commands': [sys.stdin.read()]}))
" <<<"$REMOTE_CMD" > "$SSM_PARAMS_FILE"

COMMAND_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "file://$SSM_PARAMS_FILE" \
  --query "Command.CommandId" --output text)
rm -f "$SSM_PARAMS_FILE"

echo "Waiting for deploy command ($COMMAND_ID) to finish..."
while true; do
  CMD_STATUS=$(aws ssm get-command-invocation --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" \
    --query "Status" --output text 2>/dev/null || echo "Pending")
  case "$CMD_STATUS" in
    Success) break ;;
    Failed|Cancelled|TimedOut)
      echo "ERROR: deploy command $CMD_STATUS. Output:" >&2
      aws ssm get-command-invocation --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" \
        --query "StandardErrorContent" --output text >&2
      exit 1
      ;;
    *) sleep 5 ;;
  esac
done
echo "App deployed and service restarted."

rm -rf "$BUILD_DIR"

echo "== Deploying frontend =="
FRONTEND_BUILD_DIR="$(mktemp -d)"
sed -e "s|__API_URL__|$API_URL|g" -e "s|__API_KEY__|$API_KEY_VALUE|g" \
  "$DEPLOY_DIR/frontend/index.html" > "$FRONTEND_BUILD_DIR/index.html"
cp "$DEPLOY_DIR/frontend/examples.html" "$FRONTEND_BUILD_DIR/examples.html"

aws s3 mb "s3://$FRONTEND_BUCKET" --region "$REGION" 2>/dev/null || true
aws s3api put-public-access-block --bucket "$FRONTEND_BUCKET" --public-access-block-configuration \
  "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"
aws s3 cp "$FRONTEND_BUILD_DIR/index.html" "s3://$FRONTEND_BUCKET/index.html"
aws s3 cp "$FRONTEND_BUILD_DIR/examples.html" "s3://$FRONTEND_BUCKET/examples.html"
aws s3 website "s3://$FRONTEND_BUCKET" --index-document index.html
aws s3api put-bucket-policy --bucket "$FRONTEND_BUCKET" --policy "$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::$FRONTEND_BUCKET/*"
  }]
}
EOF
)"
rm -rf "$FRONTEND_BUILD_DIR"

echo ""
echo "== Done =="
echo "Frontend: http://$FRONTEND_BUCKET.s3-website-$REGION.amazonaws.com"
echo "API:      $API_URL"
echo "Swagger:  http://$PUBLIC_IP:5000/apidocs/"
echo "API key:  $API_KEY_VALUE"
echo ""
echo "Note: the API key is embedded in the frontend page source (visible via"
echo "view-source) -- this deters casual bot scanning, it is not real auth."
echo "The instance runs 24/7 until you run ./deploy/teardown.sh -- see"
echo "docs/deployment.md for the cost this incurs."
