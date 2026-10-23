# Pandora science agent: web UI + pipeline. The same image also runs the sandbox runner
# (see docker-compose.yml). Builds for amd64 and arm64 (e.g. DGX Spark).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PANDORA_DATA_DIR=/data/pandora-data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY benchmarks ./benchmarks
COPY .claude ./.claude

# Editable install keeps prompts and skills as plain files under /app, so the admin can edit
# them in Pandora; their version history lives in the /data volume.
RUN pip install -e . \
 && useradd --uid 10001 --create-home pandora \
 && mkdir -p /data \
 && chown -R pandora /data /app/src/science_agent/prompts /app/.claude

USER pandora
EXPOSE 8765
CMD ["python", "-m", "science_agent.web", "--host", "0.0.0.0", "--port", "8765", "--no-browser", "--config", "/config/config.yaml"]
