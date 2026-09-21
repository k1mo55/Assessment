#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: ./orchestrate.sh --action start|terminate"
}

ACTION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --action)
            if [[ $# -lt 2 ]]; then
                echo "Error: --action requires a value." >&2
                usage >&2
                exit 2
            fi
            ACTION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is required but was not found." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Error: Docker Compose v2 is required." >&2
    exit 1
fi

case "$ACTION" in
    start)
        docker compose up --detach --build
        ;;
    terminate)
        docker compose down --volumes --remove-orphans
        ;;
    "")
        echo "Error: --action is required." >&2
        usage >&2
        exit 2
        ;;
    *)
        echo "Error: unsupported action: $ACTION" >&2
        usage >&2
        exit 2
        ;;
esac
