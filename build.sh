#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Running from $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# CI authenticates reads and uploads through Prefix Repository Access (OIDC).
UPLOAD="${1:---upload}"

micromamba install -y -c conda-forge python=3.11 'rattler-build>=0.80.0' requests tqdm
eval "$(micromamba shell hook -s bash)"
micromamba activate
if [ "${GITHUB_ACTIONS:-}" == "true" ]; then
    # Mint on the API host before reading repodata from repo.prefix.dev.
    PREFIX_AUTH_DIR=$(umask 077; mktemp -d)
    trap 'rm -rf -- "$PREFIX_AUTH_DIR"' EXIT
    export RATTLER_AUTH_FILE="$PREFIX_AUTH_DIR/credentials.json"
    rattler-build auth login https://prefix.dev --workload-identity
fi
rattler-build --config-file "$SCRIPT_DIR/rattler-build.toml" build \
    -c https://repo.prefix.dev/graphext -c conda-forge \
    --experimental \
    --channel-priority disabled \
    --output-dir ~/.rattler-build/ \
    --recipe recipe.yaml

if [ "$UPLOAD" == "--upload" ]; then
    LATEST_PACKAGE=$(ls -t ~/.rattler-build/noarch/cuery-*.conda | head -n 1)
    if [ "${GITHUB_ACTIONS:-}" == "true" ]; then
        # Stored workload-identity bearer tokens do not refresh automatically.
        rattler-build auth login https://prefix.dev --workload-identity
    fi
    env -u PREFIX_API_KEY rattler-build upload prefix -c graphext "$LATEST_PACKAGE"
    env -u PREFIX_API_KEY rattler-build upload prefix -c graphext/cuery "$LATEST_PACKAGE"
else
    echo "Upload disabled, skipping upload."
fi
