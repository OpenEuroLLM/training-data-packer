# Tokenize

The packer includes a tokenization step, which is documented [here](lumi.md).
That step works well when you run the full packer pipeline, but it can be cumbersome
when you just want to evaluate data mixes.

For that use case, there is a standalone tool called `oellm-tokenize`. You can run it
on your local machine (tested on macOS and Linux) or on LUMI via a Singularity container.

It tokenizes either a single file or every file in a directory. It supports the following file formats:

* `.jsonl.zst`
* `.jsonl.zstd`
* `.jsonl.gz`
* `.jsonl`
* `.parquet`

Referens to tokenizer can either be a path or a reference to Huggingface.

## LUMI

To show usage instructions:

```
singularity exec \
  /scratch/project_465002530/containers/training-data-packer/training-data-packer.sif \
  oellm-tokenize --help
```

The following example tokenizes a file with the same tokenizer used for prelude and flag.
It writes the output to the directory `output`.

```
singularity exec \
  --bind /appl/local/openeurollm \
  --bind /pfs/lustrep2/scratch \
  --bind /pfs/lustrep3/scratch \
  --bind /scratch \
  --bind /project \
  /scratch/project_465002530/containers/training-data-packer/training-data-packer.sif \
  oellm-tokenize \
  -t /scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k \
  -o output \
  /scratch/project_465002530/datasets/additional_datasets/danish-dynaword/source/data/ft/data.parquet
```

To use a tokenizer from Hugging Face, set `HF_HOME=/scratch/project_465002530/cache/huggingface`.

## Local

You need `uv` installed; see the [uv homepage](https://docs.astral.sh/uv/).

1. Clone this repository.
2. Run `uv sync` to install the dependencies.
3. Run `uv run oellm-tokenize --help` to show usage instructions.
4. Run `uv run oellm-tokenize -t openeurollm/tokenizer-256k-v2 path/to/my/data` to tokenize your data.
