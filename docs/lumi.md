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

### Hugging Face cache

`--token-count` loads tokenizers through transformers, which downloads them to `~/.cache/huggingface`.
The home directory on LUMI is small, so put the cache in the project scratch area instead:

```shell
export HF_HOME=/scratch/project_465002530/cache/huggingface
```

Compute nodes may not have internet access. If a job fails to download a tokenizer, run the same command once on a
login node to fill the cache first.

### SLURM jobs

In the repository under `tools/slurm` there are files to schedule the packer as SLURM jobs.

Note! The linter is not aimed to run with SLURM, instead it is a tool to avoid you put broken data on queue.

If the dataset shall be sampled (up or down) based on metadata such as propella, WDS or register schedule a sample step.

```
sbatch --array=0-10 "tools/slurm/package.sh" ${COLLECTION_DIR} sample
```

Once the sample is done, update budget and rubber in metadata.

The remaining three steps can run sequentially in one go using `pipeline.sh`. The steps are:
* release - Mask PII, remove contaminated data and adjust to budget.
* merge - Reduce the number of files to wanted shard size.
* tokenize - Tokenize the data.

To avoid scheduling the full pipeline with faulty metadata, the linter is running before the scheduling occur.

```
./tools/slurm/pipeline.sh ${COLLECTION_DIR} ${PACKAGE_ARRAY} ${MERGE_ARRAY} ${TOKENIZE_ARRAY}
```

Example:
```
./tools/slurm/pipeline.sh \
  /scratch/project_465002530/users/magnrune/data/training-data-collection/flag/olmo-mix-1124 \
  0-4 \
  0-1 \
  0-1
```

Logs are ending up in the sub-directory logs to your current directory. In the logs directory a `jobs.csv`
is created by `pipeline.sh` to collection and job id scheduled.



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
   $REPO_DIR/tools/convert_singularity.sh
   ```

   If the package on `ghcr.io` is private, set `SINGULARITY_DOCKER_USERNAME` and `SINGULARITY_DOCKER_PASSWORD`
   (a GitHub token with `read:packages`) before you pull.

3. Copy the wrapper to the shared directory, if it is new or has changed:

   ```shell
   cp tools/oellm-package-data.sh /scratch/project_465002530/containers/training-data-packer/
   chmod g+rx /scratch/project_465002530/containers/training-data-packer/oellm-package-data.sh
   ```
