# Run on LUMI

On LUMI the packer runs from a Singularity container, so you don't need `uv` or a Python setup on the cluster.
The container and a wrapper script are installed in a shared directory of project `project_465002530`:

```
/scratch/project_465002530/containers/training-data-packer/
    oellm-package-data.sh       wrapper that runs oellm-package-data in the container
    training-data-packer.sif    the container image
```

See [containers.md](containers.md) for what the image contains.

## Run the packer

`oellm-package-data.sh` takes the same arguments as `oellm-package-data`:

```shell
PACKER=/scratch/project_465002530/containers/training-data-packer/oellm-package-data.sh

$PACKER --help
$PACKER --collection-dir <dir> --mode lint
```

The wrapper binds these paths into the container: `/scratch`, `/project`, `/appl/local/openeurollm`,
`/pfs/lustrep2/scratch` and `/pfs/lustrep3/scratch`. Your home directory is also available.
The collection directory, and every file `metadata.yaml` refers to, must be under one of these paths.

The container runs as your own user and gets your environment, so variables such as `LOGURU_LEVEL`
and `HF_HOME` work as they do outside the container.

### Interactive use on a login node

Lint mode is light and fast, so you can run it directly on a login node:

```shell
$PACKER --collection-dir /scratch/project_465002530/<dataset> --mode lint
```

Don't run `release`, `sample` or `merge` on a login node. Submit them as SLURM jobs.

### SLURM jobs

Create a batch script that calls the wrapper, for example `package-container.sh`:

```shell
#!/bin/bash
#SBATCH --partition=small
#SBATCH --job-name=package
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/package_%A_%a.out
#SBATCH --error=logs/package_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

export LOGURU_LEVEL=INFO

time /scratch/project_465002530/containers/training-data-packer/oellm-package-data.sh \
    --collection-dir "$1" --slurm --workers "${SLURM_CPUS_PER_TASK}" "${@:2}"
```

Submit it with the collection directory and any extra arguments:

```shell
mkdir -p logs
sbatch package-container.sh /scratch/project_465002530/<dataset>                # release mode
sbatch package-container.sh /scratch/project_465002530/<dataset> --mode merge   # merge mode
```

Set `--cpus-per-task` to the number of workers you want, since the script passes it on as `--workers`.

### Hugging Face cache

`--token-count` loads tokenizers through transformers, which downloads them to `~/.cache/huggingface`.
The home directory on LUMI is small, so put the cache in the project scratch area instead:

```shell
export HF_HOME=/scratch/project_465002530/cache/huggingface
```

Compute nodes may not have internet access. If a job fails to download a tokenizer, run the same command once on a
login node to fill the cache first.

## Install or update the container

Maintainers do this when a new version of the packer is released. Users only need the steps above.

1. Build the image for `amd64` and `arm64` and push it to the GitHub container registry.
   On a Mac with Apple's `container` tool:

   ```shell
   tools/build_container.sh
   ```

   This pushes `ghcr.io/openeurollm/training-data-packer:latest`. You need to be logged in to `ghcr.io` with
   permission to push to the `openeurollm` organization.

2. On a LUMI login node, pull the image and convert it to a SIF file in the shared directory:

   ```shell
   cd /scratch/project_465002530/containers/training-data-packer
   singularity pull --force training-data-packer.sif docker://ghcr.io/openeurollm/training-data-packer:latest
   ```

   If the package on `ghcr.io` is private, set `SINGULARITY_DOCKER_USERNAME` and `SINGULARITY_DOCKER_PASSWORD`
   (a GitHub token with `read:packages`) before you pull.

3. Copy the wrapper to the shared directory, if it is new or has changed:

   ```shell
   cp tools/oellm-package-data.sh /scratch/project_465002530/containers/training-data-packer/
   chmod g+rx /scratch/project_465002530/containers/training-data-packer/oellm-package-data.sh
   ```
