#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Running from $SCRIPT_DIR"
cd "$SCRIPT_DIR"

TOKEN_FILE="$1"
# Set the UPLOAD variable to the second command-line argument ($2) if provided,
# otherwise default to "--upload" if no second argument is given.
# This allows the script to be called with or without specifying an upload flag.
UPLOAD="${2:---upload}"

micromamba install -y -c conda-forge python=3.11 rattler-build requests tqdm
rattler-build auth login https://repo.prefix.dev --token $(cat "$TOKEN_FILE")
rattler-build build \
    -c https://repo.prefix.dev/graphext -c conda-forge \
    --experimental \
    --channel-priority disabled \
    --output-dir ~/.rattler-build/ \
    --recipe recipe.yaml

if [ "$UPLOAD" == "--upload" ]; then
    LATEST_PACKAGE=$(ls -t ~/.rattler-build/noarch/cuery-*.conda | head -n 1)
    rattler-build upload prefix -c graphext "$LATEST_PACKAGE"
else
    echo "Upload disabled, skipping upload."
fi
