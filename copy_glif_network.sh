#!/usr/bin/env bash
# Copy biorealistic-v1-model-latest/core_<suffix>/network into
# V1_GLIF_model_tf_training/GLIF_network_<suffix>.
#
# Usage:
#   ./copy_glif_network.sh                  # copy all core_* variants
#   ./copy_glif_network.sh -s <suffix>      # copy one specific variant

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATEST="$(dirname "$SCRIPT_DIR")/biorealistic-v1-model-latest"

usage() {
    echo "Usage: $0 [-s <network_suffix>]"
    echo "  -s  Variant suffix (e.g. L4-e2e-syn-as-L6). Omit to copy all core_* variants."
    exit 1
}

SUFFIX=""
while getopts ":s:h" opt; do
    case $opt in
        s) SUFFIX="$OPTARG" ;;
        h) usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
    esac
done

copy_one() {
    local suffix="$1"
    local src="$LATEST/core_${suffix}/network"
    local dest="$SCRIPT_DIR/GLIF_network_${suffix}"

    if [[ ! -d "$src" ]]; then
        echo "WARNING: source not found: $src" >&2
        return 1
    fi

    if [[ -d "$dest" ]]; then
        echo "SKIP (already exists): GLIF_network_${suffix}"
        return 0
    fi

    cp -r "$src" "$dest"
    echo "DONE: GLIF_network_${suffix}"
}

if [[ -n "$SUFFIX" ]]; then
    copy_one "$SUFFIX"
else
    found=0
    for core_dir in "$LATEST"/core_*/; do
        variant="${core_dir%/}"
        variant="${variant##*/core_}"
        if [[ -d "$core_dir/network" ]]; then
            copy_one "$variant"
            ((found++)) || true
        else
            echo "SKIP (no network subdir): core_${variant}" >&2
        fi
    done
    echo "Done. Processed $found variant(s)."
fi
