#!/bin/sh
# Install or update the public demo web app on the stack's Graviton worker, through SSM (no SSH, no keys).
# Usage: make web-up   (uploads the code first). The URL is the stack output WebUrl.
set -eu
WORKER=$(aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='WorkerId'].OutputValue" --output text)
SCRIPT=$(cat <<SH
set -e
cd /opt/gaugeless
aws s3 cp s3://$BUCKET/code/gaugeless.tar.gz code.tar.gz --only-show-errors
tar xzf code.tar.gz
PY=\$(systemctl show gaugeless -p ExecStart --value | sed -n 's/.*path=\([^ ;]*\).*/\1/p')
cat > /etc/systemd/system/gaugeless-web.service <<UNIT
[Unit]
Description=Gaugeless web
After=network-online.target
[Service]
WorkingDirectory=/opt/gaugeless
Environment=AWS_REGION=$REGION AWS_DEFAULT_REGION=$REGION
ExecStart=\$PY -m gaugeless.web.server --host 0.0.0.0 --port 80 --public
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable gaugeless-web
systemctl restart gaugeless gaugeless-web
sleep 3
systemctl is-active gaugeless gaugeless-web
SH
)
PARAMS=$(python3 -c 'import json,sys; print(json.dumps({"commands": sys.stdin.read().splitlines()}))' <<EOF2
$SCRIPT
EOF2
)
ID=$(aws ssm send-command --instance-ids "$WORKER" --document-name AWS-RunShellScript --parameters "$PARAMS" \
  --query Command.CommandId --output text)
aws ssm wait command-executed --command-id "$ID" --instance-id "$WORKER" || true
aws ssm get-command-invocation --command-id "$ID" --instance-id "$WORKER" \
  --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
