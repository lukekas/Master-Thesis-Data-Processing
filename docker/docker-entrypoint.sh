#!/usr/bin/env sh
set -eu

uv run python -m poi_data
uv run python -m graph_modeling
uv run python -m hexagon_grid
uv run fastapi run backend.py --host=0.0.0.0 --port 80