#!/bin/bash
cd "$(dirname "$0")" || exit
rm -r dist/
uv build
uv publish --token "$(cat ~/Development/config/pypi-publish-token.txt)"