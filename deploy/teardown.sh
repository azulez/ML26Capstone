#!/usr/bin/env bash
# Removes everything deploy.sh created: the CloudFormation stack (EC2
# instance, security group, IAM role, Elastic IP), the code S3 bucket, the
# API key SSM parameter, and the frontend S3 bucket. Run this once the
# instance is no longer needed -- unlike the old Lambda backend, it bills
# 24/7 until torn down.
set -euo pipefail

STACK_NAME="ml26-fintc"
CODE_BUCKET="${CODE_BUCKET:-ml2026-earnings-call-code}"
FRONTEND_BUCKET="${FRONTEND_BUCKET:-ml2026-earnings-call-frontend}"
API_KEY_PARAM="/ml26-fintc/api-key"

echo "== Deleting CloudFormation stack (EC2 instance, security group, IAM role, EIP) =="
aws cloudformation delete-stack --stack-name "$STACK_NAME"
aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME"

echo "== Deleting code bucket =="
aws s3 rb "s3://$CODE_BUCKET" --force 2>/dev/null || echo "Bucket $CODE_BUCKET already gone or never created."

echo "== Deleting API key parameter =="
aws ssm delete-parameter --name "$API_KEY_PARAM" 2>/dev/null || echo "Parameter $API_KEY_PARAM already gone or never created."

echo "== Deleting frontend bucket =="
aws s3 rb "s3://$FRONTEND_BUCKET" --force 2>/dev/null || echo "Bucket $FRONTEND_BUCKET already gone or never created."

echo "== Done =="
echo "If you enabled the billing alarm, its SNS topic/subscription and the"
echo "CloudWatch alarm were removed with the stack. Double-check the AWS"
echo "console (EC2, S3, CloudFormation, Systems Manager) for anything left"
echo "over before considering this fully torn down."
