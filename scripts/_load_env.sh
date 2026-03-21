#!/bin/bash
# Internal helper — source this file to load .env safely.
# Handles: leading/trailing whitespace around values, inline comments,
# blank lines, and special characters in passwords.
# Usage (in other scripts):  source "$(dirname "${BASH_SOURCE[0]}")/_load_env.sh"

_load_dotenv() {
    local env_file="${1}"
    if [[ ! -f "${env_file}" ]]; then
        echo "ERROR: .env not found at ${env_file}" >&2
        exit 1
    fi

    while IFS= read -r line || [[ -n "${line}" ]]; do
        # Skip blank lines and comment lines
        [[ "${line}" =~ ^[[:space:]]*$ ]]  && continue
        [[ "${line}" =~ ^[[:space:]]*#  ]] && continue

        # Split on the first = only
        local key="${line%%=*}"
        local val="${line#*=}"

        # Trim leading whitespace from value
        val="${val#"${val%%[![:space:]]*}"}"

        # Strip optional surrounding single or double quotes
        if [[ "${val}" =~ ^\"(.*)\"$ ]]; then
            val="${BASH_REMATCH[1]}"
        elif [[ "${val}" =~ ^\'(.*)\'$ ]]; then
            val="${BASH_REMATCH[1]}"
        fi

        # Trim trailing whitespace
        val="${val%"${val##*[![:space:]]}"}"

        # Export into the calling shell's environment
        export "${key}=${val}"
    done < "${env_file}"
}
