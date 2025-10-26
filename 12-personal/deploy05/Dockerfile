# FROM python:3.11.13 # -slim-bookworm
FROM agrigorev/zoomcamp-model:2025

# RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin/:$PATH"

COPY ".python-version" "pyproject.toml" "uv.lock" ./
RUN uv sync --locked

COPY "predict6.py" "./"

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict6:app", "--host", "0.0.0.0", "--port", "9696"]
