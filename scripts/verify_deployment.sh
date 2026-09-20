#!/usr/bin/env bash
# Capture Checkpoint 4 deployment evidence.
#
#   ./scripts/verify_deployment.sh
#
# Builds the image, starts the container, waits for the healthcheck, exercises
# the CLI entrypoints, and writes everything to docs/deployment_evidence.txt.
#
# Run this on a machine with unrestricted Docker access. It could not be run in
# the development environment, where Docker Hub's image-blob CDN is blocked by
# network policy - see docs/05_Deployment.md section 5.
set -uo pipefail

EVIDENCE="docs/deployment_evidence.txt"
PORT="${PORT:-8501}"
TIMEOUT_SECS="${TIMEOUT_SECS:-180}"

cd "$(dirname "$0")/.."

# Everything this script prints is also appended to the evidence file, so the
# captured artifact matches exactly what the operator saw.
exec > >(tee "$EVIDENCE") 2>&1

echo "===================================================================="
echo "DEPLOYMENT EVIDENCE - Personal Assistant AI"
echo "===================================================================="
echo "Captured : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Host     : $(uname -srm)"
echo "Docker   : $(docker --version 2>/dev/null || echo 'NOT AVAILABLE')"
echo "Compose  : $(docker compose version 2>/dev/null || echo 'NOT AVAILABLE')"
echo "Commit   : $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

if ! docker info >/dev/null 2>&1; then
  echo
  echo "FAIL: the Docker daemon is not reachable. Start Docker and re-run."
  exit 1
fi

echo
echo "--- 1. BUILD -------------------------------------------------------"
if ! docker compose build; then
  echo "FAIL: image build failed. See the output above."
  exit 1
fi
echo "PASS: image built"

echo
echo "--- 2. IMAGE -------------------------------------------------------"
docker images personal-assistant-ai --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'

echo
echo "--- 3. START -------------------------------------------------------"
docker compose up -d
sleep 5
docker compose ps

echo
echo "--- 4. HEALTHCHECK -------------------------------------------------"
echo "Waiting up to ${TIMEOUT_SECS}s for the container to report healthy..."
deadline=$(( SECONDS + TIMEOUT_SECS ))
status="unknown"
while [ $SECONDS -lt $deadline ]; do
  cid=$(docker compose ps -q app 2>/dev/null | head -1)
  if [ -n "$cid" ]; then
    status=$(docker inspect --format '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo "no-healthcheck")
    [ "$status" = "healthy" ] && break
    [ "$status" = "unhealthy" ] && break
  fi
  sleep 5
done
echo "Container health: $status"

echo
echo "--- 5. HTTP RESPONSE -----------------------------------------------"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/_stcore/health" || echo "000")
body=$(curl -s "http://localhost:${PORT}/_stcore/health" || echo "(no response)")
echo "GET /_stcore/health -> HTTP ${code}, body: ${body}"
root=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/" || echo "000")
echo "GET /                -> HTTP ${root}"

echo
echo "--- 6. CLI ENTRYPOINTS ---------------------------------------------"
echo "\$ docker compose run --rm app reminders"
docker compose run --rm app reminders 2>&1 | head -20

echo
echo "--- 7. PERSISTENCE -------------------------------------------------"
echo "Named volumes (the index must survive container replacement):"
docker volume ls --filter name=assistant --format 'table {{.Name}}\t{{.Driver}}'

echo
echo "--- 8. TEARDOWN ----------------------------------------------------"
docker compose down
echo "Containers stopped. Volumes retained."

echo
echo "===================================================================="
if [ "$status" = "healthy" ] && [ "$code" = "200" ]; then
  echo "RESULT: PASS - image builds, container serves, healthcheck green."
  echo "Evidence written to ${EVIDENCE}"
  exit 0
fi
echo "RESULT: FAIL - health=${status}, HTTP=${code}. See output above."
echo "===================================================================="
exit 1
