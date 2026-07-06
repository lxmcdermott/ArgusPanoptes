# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Argus Panoptes - placeholder image (Day 6 deliverable).
# This is a SCAFFOLD: it installs the core dependencies so the sensors/ module
# and dataset generation run in-container. The API + dashboard entrypoints are
# added once app/ is implemented.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run simulator validation. Replace with API/dashboard entrypoint later.
CMD ["python", "scripts/validate_simulators.py", "--no-show"]
