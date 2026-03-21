#!/bin/bash
# Sync Snellius results back to local machine.
# Credentials read exclusively from .env — never hardcoded here.
# Usage: bash scripts/sync_results_back.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/_load_env.sh"
_load_dotenv "${PROJECT_ROOT}/.env"

for var in SNELLIUS_HOST SNELLIUS_USER SNELLIUS_PASSWORD SNELLIUS_PORT SNELLIUS_REMOTE_DIR; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: ${var} is not set in .env" >&2; exit 1
    fi
done

export SSHPASS="${SNELLIUS_PASSWORD}"

LOCAL_RESULTS="${PROJECT_ROOT}/results"
mkdir -p "${LOCAL_RESULTS}"

echo "Syncing [host]:[remote dir]/results/ → ${LOCAL_RESULTS}/"

sshpass -e rsync -avz --progress \
    -e "ssh -p ${SNELLIUS_PORT} -o StrictHostKeyChecking=accept-new" \
    "${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_REMOTE_DIR}/results/" \
    "${LOCAL_RESULTS}/"

unset SSHPASS
echo "Sync complete. Results in: ${LOCAL_RESULTS}"
