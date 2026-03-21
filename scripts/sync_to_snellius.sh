#!/bin/bash
# Sync local NODIS project to Snellius HPC.
# Credentials read exclusively from .env — never hardcoded here.
# Usage: bash scripts/sync_to_snellius.sh

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

echo "Syncing ${PROJECT_ROOT}/ → [host]:[remote dir]"

sshpass -e rsync -avz --progress \
    -e "ssh -p ${SNELLIUS_PORT} -o StrictHostKeyChecking=accept-new" \
    --exclude ".venv/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "*.egg-info/" \
    --exclude ".pytest_cache/" \
    --exclude ".env" \
    --exclude "data/" \
    --exclude "results/" \
    --exclude "output/" \
    --exclude "logs/" \
    --exclude ".git/" \
    "${PROJECT_ROOT}/" \
    "${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_REMOTE_DIR}/"

unset SSHPASS
echo "Sync complete."
