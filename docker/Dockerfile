FROM ghcr.io/lukekas/master-thesis-frontend:latest AS frontend

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV GTFS_PATH=/app/data/GTFS
ENV DGM_PATH=/app/data/dgm/downloads
ENV DSM_PATH=/app/data/dsm/downloads
ENV FRONTEND_PATH=/app/frontend
ENV OSM_TAGS_PATH=/app/tags
ENV LCZ_PATH=/app/data/lcz/lcz_v3.tif

WORKDIR /app
COPY src/* /app/
COPY pyproject.toml /app/
COPY uv.lock /app/
COPY data/tags /app/tags
COPY data/evaluation/*.csv /app/data/evaluation/
COPY data/lcz /app/data/lcz
COPY docker/docker-entrypoint.sh /app/docker-entrypoint.sh
COPY --from=frontend /usr/share/nginx/html /app/frontend

RUN apt-get update && apt-get install -y --no-install-recommends libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN chmod +x /app/docker-entrypoint.sh
RUN mkdir -p /app/results
RUN mkdir -p /app/cache


RUN uv sync --locked
CMD ["./docker-entrypoint.sh"]