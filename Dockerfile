FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/vascumamba-spot
COPY pyproject.toml README.md LICENSE CITATION.bib ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install "numpy<2" torch --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install -e ".[cuda,dev]"

CMD ["pytest"]
