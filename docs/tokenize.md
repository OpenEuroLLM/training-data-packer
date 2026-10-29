# Tokenize

The packer has a built-in tokenization step, described in [LUMI](lumi.md).
That step works well as part of the full packer pipeline, but it is cumbersome
when you only want to evaluate data mixes.

For that use case, use the standalone tool `oellm-tokenize-standalone`. You can run it
on your local machine (tested on macOS and Linux) or on LUMI through a Singularity container.

The tool tokenizes a single file or every file in a directory. It supports these file formats:

* `.jsonl.zst`
* `.jsonl.zstd`
* `.jsonl.gz`
* `.jsonl`
* `.parquet`

You can give the tokenizer as a local path or as a Hugging Face Hub repository ID.

The tool is opinionated: it adds a `<bos>` token at the start of each document and an `<eos>` token at the end.
Use the `--no-bos` flag to leave out the `<bos>` token.

## LUMI

To show usage instructions:

```shell
singularity exec \
  /scratch/project_465002530/containers/training-data-packer/training-data-packer.sif \
  oellm-tokenize-standalone --help
```

The following example tokenizes a file with the tokenizer used for Prelude and Flag,
and writes the output to the directory `output`:

```shell
singularity exec \
  --bind /appl/local/openeurollm \
  --bind /pfs/lustrep2/scratch \
  --bind /pfs/lustrep3/scratch \
  --bind /scratch \
  --bind /project \
  /scratch/project_465002530/containers/training-data-packer/training-data-packer.sif \
  oellm-tokenize-standalone \
  -t /scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k \
  -o output \
  /scratch/project_465002530/datasets/additional_datasets/danish-dynaword/source/data/ft/data.parquet
```

To use a tokenizer from Hugging Face, set `HF_HOME=/scratch/project_465002530/cache/huggingface`.

To spread the files over a SLURM job array, add `--slurm`. Each array task starts at the file given by
`SLURM_ARRAY_TASK_ID` and then tokenizes every `SLURM_ARRAY_TASK_COUNT`-th file.
The SLURM script `tokenize-standalone.sh` handles the scheduling for you:

```shell
sbatch --array=0-3 /scratch/project_465002530/containers/training-data-packer/tokenize-standalone.sh \
  -t /scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k \
  --text text \
  -o output \
  /scratch/project_465002530/training/collection/flag/starcoder-0.0.0/release
```

## Local

You need `uv` installed; see the [uv homepage](https://docs.astral.sh/uv/).

1. Clone this repository.
2. Run `uv sync` to install the dependencies.
3. Run `uv run oellm-tokenize-standalone --help` to show usage instructions.
4. Run `uv run oellm-tokenize-standalone -t openeurollm/tokenizer-256k-v2 path/to/my/data` to tokenize your data.
