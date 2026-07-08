# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Argus Panoptes - runtime image for the FastAPI service and the NiceGUI
# operator dashboard (and one-shot dataset generation). Installs the package
# with the serving + dashboard extras so any of the compose services can run.
# See deployment/docker-compose.yml.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (better layer caching) using the package metadata.
COPY pyproject.toml requirements.txt ./
COPY sensors ./sensors
COPY dsp ./dsp
COPY models ./models
COPY app ./app
COPY dashviz ./dashviz
RUN pip install --no-cache-dir -e ".[ml,dl,app,dashboard-nicegui]"

# Bring in the rest (scripts, experiments artifacts, configs, etc.).
COPY . .

EXPOSE 8000 8080

# Default entrypoint is the API service; the dashboard service overrides the
# command (see docker-compose.yml). Bind 0.0.0.0 so the port maps out.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
