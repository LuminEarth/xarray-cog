FROM rust:1.91-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/build && /opt/build/bin/pip install 'maturin>=1.9,<2'
WORKDIR /src
COPY . .
RUN /opt/build/bin/maturin build --release --locked --out /wheels

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir 'pip>=25.1' \
    && pip install --no-cache-dir /wheels/*.whl --group benchmark \
    && rm -rf /wheels /app/pyproject.toml
COPY scripts/benchmark.py /app/scripts/benchmark.py
ENV BENCH_MODE=docker
ENTRYPOINT ["python", "/app/scripts/benchmark.py"]
