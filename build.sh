#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Running from $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# CI authenticates reads and uploads through Prefix Repository Access (OIDC).
UPLOAD="${1:---upload}"

micromamba install -y -c conda-forge python=3.11 rattler-build requests tqdm
eval "$(micromamba shell hook -s bash)"
micromamba activate
CUTOFF=$(python -c 'from datetime import datetime, timedelta, timezone; print((datetime.now(timezone.utc) - timedelta(days=7)).isoformat())')
rattler-build build \
    -c https://repo.prefix.dev/graphext -c conda-forge \
    --experimental \
    --channel-priority disabled \
    --exclude-newer "$CUTOFF" \
    --output-dir ~/.rattler-build/ \
    --recipe recipe.yaml

if [ "$UPLOAD" == "--upload" ]; then
    LATEST_PACKAGE=$(ls -t ~/.rattler-build/noarch/cuery-*.conda | head -n 1)
    env -u PREFIX_API_KEY rattler-build upload prefix -c graphext "$LATEST_PACKAGE"
    env -u PREFIX_API_KEY rattler-build upload prefix -c graphext/cuery "$LATEST_PACKAGE"
else
    echo "Upload disabled, skipping upload."
fi
