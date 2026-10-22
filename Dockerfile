# Container image for the oellm-* tools.
#
# Build: docker build -t training-data-packer .
# Run:   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:$PWD" -w "$PWD" \
#            training-data-packer oellm-package-data --collection-dir <dir> --mode lint
#
# The virtual environment lives in /opt/venv, the same path tools/run_in_container.sh mounts it at.

ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /src

# Install the dependencies first, so they are cached as long as the lock file is unchanged.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY training_data_packer ./training_data_packer
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:${PYTHON_VERSION}-slim

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home packer
USER packer

CMD ["oellm-package-data", "--help"]
