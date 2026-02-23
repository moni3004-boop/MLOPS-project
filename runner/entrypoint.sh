#!/bin/bash
set -euo pipefail

api="https://api.github.com"
owner="${GH_OWNER}"
repo="${GH_REPO}"

reg_json="$(curl -fsSL -X POST \
  -H "Authorization: token ${GH_PAT}" \
  -H "Accept: application/vnd.github+json" \
  "${api}/repos/${owner}/${repo}/actions/runners/registration-token")"

reg_token="$(echo "$reg_json" | sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
if [ -z "$reg_token" ]; then
  echo "Failed to parse registration token. Response:"
  echo "$reg_json"
  exit 1
fi

./config.sh \
  --url "https://github.com/${owner}/${repo}" \
  --token "${reg_token}" \
  --name "${RUNNER_NAME}" \
  --labels "${RUNNER_LABELS}" \
  --unattended \
  --replace

exec ./run.sh