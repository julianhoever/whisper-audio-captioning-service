FROM ghcr.io/astral-sh/uv:trixie-slim

RUN apt-get update && apt-get install -y curl build-essential libz-dev libxml2-dev libxslt-dev python3-lxml

WORKDIR /app
COPY . .
RUN uv sync --no-dev --locked

CMD ["uv", "run", "--no-dev", "uvicorn", "--host=0.0.0.0", "--port=80", "audiocap.main:app"]
