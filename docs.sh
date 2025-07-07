#!/bin/bash
cd docs && uv run make clean html && cd build/html && uv run python -m http.server