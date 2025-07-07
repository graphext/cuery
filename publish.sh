#!/bin/bash
rm -rI dist/
uv build
uv publish --token "$(cat ~/Development/config/pypi-publish-token.txt)"