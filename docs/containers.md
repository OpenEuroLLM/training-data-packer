# Containers

The `Dockerfile` in the repository root builds an image with the four `oellm-*` tools installed.
Use it to run the packer without a local Python and `uv` setup.

## Image contents

- Base image: `python:3.14-slim`.
- The dependencies come from `uv.lock` (`uv sync --locked --no-dev`), so the image uses the same versions as a
  local `uv sync`. Dev dependencies and the `megatron` extra (torch) are not installed.
  When a tool starts, transformers prints `PyTorch was not found`. This is expected and harmless.
- The tools are installed in a virtual environment in `/opt/venv`, which is on `PATH`.
- The image runs as the non-root user `packer`. With no command, it prints `oellm-package-data --help`.

Only `pyproject.toml`, `uv.lock`, `README.md` and `training_data_packer/` are sent to the build
(see `.dockerignore`), so local data, `.venv` and test files never end up in the image.

## Build

From the repository root:

```shell
docker build -t training-data-packer .
```

Podman works the same way: replace `docker` with `podman`.

Docker caches the dependency layer, so a rebuild after a code change only reinstalls the project itself.
Changing `uv.lock` or `pyproject.toml` reinstalls all dependencies.

To build for a different architecture, for example an `x86_64` cluster from an Apple silicon Mac,
add `--platform linux/amd64`:

```shell
docker build --platform linux/amd64 -t training-data-packer .
```

## Run

Mount the collection directory into the container and pass the tool and its arguments as the command:

```shell
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$PWD:$PWD" -w "$PWD" \
    training-data-packer \
    oellm-package-data --collection-dir <dir> --mode lint
```

- `-v "$PWD:$PWD" -w "$PWD"` mounts the current directory at the same path inside the container.
  Relative paths and absolute paths under the current directory then work as they do on the host.
  Mount every other directory the tool reads or writes, for example the collection directory or the
  PII and contamination data, if it is outside the current directory.
- `--user "$(id -u):$(id -g)"` makes the files the tool writes owned by you instead of by the `packer` user.
- The other tools run the same way:

  ```shell
  docker run --rm training-data-packer oellm-collect-metrics --help
  docker run --rm training-data-packer oellm-propella-structure --help
  docker run --rm training-data-packer oellm-propella-merge --help
  ```

### Hugging Face cache

Token counting loads tokenizers through transformers, which caches downloads in `~/.cache/huggingface`.
When you run with `--user`, the container has no writable home directory, so mount a cache directory and
point `HF_HOME` at it:

```shell
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$PWD:$PWD" -w "$PWD" \
    -v "$HOME/.cache/huggingface:/hf-cache" -e HF_HOME=/hf-cache \
    training-data-packer \
    oellm-package-data --collection-dir <dir> --token-count
```

## Run with Singularity or Apptainer

HPC clusters such as LUMI run Singularity or Apptainer instead of Docker.
For the ready-made setup on LUMI, see [lumi.md](lumi.md).
Export the image to an archive and convert it to a SIF file:

```shell
docker build --platform linux/amd64 -t training-data-packer .
docker save -o training-data-packer.tar training-data-packer
singularity build training-data-packer.sif docker-archive://training-data-packer.tar
```

Copy the SIF file to the cluster and run a tool with `singularity exec`. Bind the directories the tool needs:

```shell
singularity exec --bind /scratch \
    training-data-packer.sif \
    oellm-package-data --collection-dir <dir> --mode lint
```

Singularity runs the container as your own user and mounts your home directory by default.
