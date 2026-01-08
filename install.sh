#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -z "$CONDA_PREFIX" ]; then
    echo -e "\n${Red}Must have an active conda environment. Quitting!${Reset}\n"
    exit 1
fi

if ! hash micromamba 2>/dev/null && ! hash mamba 2>/dev/null && [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo -e "\n${Red}You probably don't want to install into the base environment! Create and/or activate a specific environment to install into.${Reset}\n"
    exit 1
fi

PREFIX_RO_TOKEN="pfx-TjYnC7XWSwSH9UMlpMPl1yCdl4e0B2h096Qt"
micromamba auth login https://repo.prefix.dev --bearer $PREFIX_RO_TOKEN

micromamba install -y -c https://repo.prefix.dev/graphext -c conda-forge cuery hatchling
micromamba remove -y -f cuery

pip install --root-user-action ignore --no-build-isolation --no-deps -e .