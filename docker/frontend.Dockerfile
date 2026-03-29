FROM ghcr.io/lukekas/master-thesis-frontend:latest AS frontend

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV FRONTEND_PATH=/app/frontend

WORKDIR /app
COPY src/* /app/
COPY pyproject.toml /app/
COPY uv.lock /app/
COPY data/evaluation/*.parquet /app/data/evaluation/
COPY results/hexagon_grid_bamberg.parquet /app/results/hexagon_grid_bamberg.parquet
COPY results/hexagon_grid_bamberg.geojson /app/results/hexagon_grid_bamberg.geojson

COPY --from=frontend /usr/share/nginx/html /app/frontend

RUN uv sync --locked
CMD ["uv", "run", "fastapi", "run", "/app/backend.py", "--port=80", "--host=0.0.0.0"]
EXPOSE 80