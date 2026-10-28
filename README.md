# training-data-packer
Packaging annotated datasets into final training data. Purpose is to have a repeatable process and
well-packaged data to remove any data management in the training step.

The packer consists of the following tools:
* `oellm-package-data` - Take source files and apply decontamination, PII-masking, and sampling. Each file
    in the source directory gets a correspondent file with the data processed. The tool is idempotent, if it fails,
    then run it again, and it will take of where it left. This is the main tool.
    The tool contain three sub tools: `lint`, `sample`, `release` (default), and `merge`.
* `oellm-collect-metrics` - Collect and summarize metrics from a collection directory.
  This does also happens automatically when run `oellm-package-data` in mode `merge`.
* `oellm-propella-structure` - Structure Propella data based on source data structure. For each record in the
    source files, if its ID exists in the Propella data, it is written to the output. This arranges Propella
    data in the same oßrder as source data.
* `oellm-propella-merge` - If Propella data is to big to be in memory `oellm-propella-structure` can run on
  individual Propella-parquet files. Then use this tool to merge the results from all Propella-parquet files.

All tools read a file `metadata.yaml` containing metadata about the structure and processing of the data.

## TL;DR; I just want to run!
Here are steps to run packaging and reduce the number of files. Instructions are intended for Lumi and using latest Singularity container.
1. Check out `https://github.com/OpenEuroLLM/training-data-packer`
2. In `tools/slurm` there are scripts to schedule jobs with SLURM.
3. Lint your metadata to find errors in the declaration: `/scratch/project_465002530/containers/training-data-packer/oellm-package-data.sh --collection-dir /scratch/project_465002530/training/collection/flag/finepdfs-edu-1.0.0 --mode lint`
4. Start pipline with package, merge, and tokenization `./tools/slurm/pipeline.sh /scratch/project_465002530/training/collection/flag/finepdfs-edu-1.0.0 0-10 0-38 0-38`
Change path to the dataset you want to process. Adjust the array size to the size of data.
5. Check logs and metrics for each step.
6. Move logs from `logs` directory within the directory where you checked out the code into `/scratch/project_465002530/training/collection/flag/log/release`

The two first steps are a one time operation or when you update the packager.


## Dataset Directory Structure

A dataset directory managed by the packager contains input sources, annotations, configuration, and generated outputs. The directories relate to each other as follows:

```text
dataset-directory/
├── metadata.yaml        # Dataset configuration, schema mappings, and pipeline rules
├── source/              # Original raw data files (e.g., .jsonl.gz), arranged in hierarchical subdirectories
├── open-privacy-filter/ # PII annotation files mirroring source/ paths (documents PII spans to mask). Output from open-privacy-filter
├── nemo-curator/        # Contamination annotation files mirroring source/ paths (documents benchmark-contaminated records to remove). Output from Nemo-Curator
├── propella-4b/         # Optional Propella quality score parquet files used for quality filtering
├── release-raw/         # Output of oellm-package-data, mode release: per-file processed data with PII masked, decontaminated, and sampled
├── release/             # Output of oellm-package-data, mode merge: consolidated and merged files ready for tokenization
├── megatron-lm/         # Output of oellm-package-data, mode tokenize: one subdirectory per tokenizer in the megatron-lm section
└── sample/              # Output of oellm-package-data, mode sample: files sampled on quality using propella-4b and other quality metrics
```

### Directory Roles and Data Flow

1. **Input & Annotations:**
   - **`source/`**: The primary input data. Files are typically organized by domain, language, or partition.
   - **`open-privacy-filter/`, `propella-4b` & `nemo-curator/`**: Annotation companion directories. Every annotation file mirrors the relative path and filename of the corresponding file in `source/`.
   - **`metadata.yaml`**: Coordinates the pipeline by specifying dataset name, source URLs, annotation file formats, sampling ratios, and shard parameters.

2. **Packaging (`oellm-package-data`):**
   - Reads `source/`, applies masking annotations from `pii/` and exclusions from `contamination/`, and writes the packaged files to **`release-raw/`**.
   - Output files in `release-raw/` preserve the directory hierarchy of `source/` and include hidden metric sidecars (`.*.metrics.json`).

3. **Consolidation (`oellm-package-data -m merge`):**
   - Reads files from **`release-raw/`** and merges smaller chunks into target-sized files in **`release/`**, maintaining path semantics while reducing file handle overhead for tokenization and pretraining.
   - Compile metrics


## Related projects
This project uses data produced from several other projects:
* [Training data collection](https://github.com/OpenEuroLLM/training-data-collection) containing the
  metadata for OpenEuroLLM models
* [Decontamination](https://github.com/OpenEuroLLM/pretraining-decontamination)
* PII detection for [baby cycle](https://github.com/OpenEuroLLM/pii-masking-oellm)
  and [flag cycle](https://github.com/OpenEuroLLM/pii-detection-solo)

After packaging is happening tokenization must happen before the training. This is
done according to the description [here](https://github.com/OpenEuroLLM/tokenizer/tree/main/baby-tokenization).

## Requirements
You must have `uv` installed, see [uv-homepage](https://docs.astral.sh/uv/).

Check out this [repo](https://github.com/OpenEuroLLM/training-data-packer#) and do:
```shell
uv sync --dev
uv run pre-commit install
```

Tip is to do an `uv sync` evry time pulling from the repository.

The software is tested on Linux and MacOS. These are the focus platforms. It is not tested on Windows.

To run the tools in a container instead, see [docs/containers.md](docs/containers.md).
To run on LUMI, see [docs/lumi.md](docs/lumi.md).

## Run

## Linter: oellm-package-data
The linter check the validity of `metadata.yaml` file and the directory structure for annoying errors.
Run the linter pefore packaging data, it will save you time.
The lint mode check:
* metadata.yaml follow schema
* Extensive checks on fields that are correlated to each other.
* Reads some data files, if exists, to verify alignment with `metadata.yaml`.

To lint `tests/resources/integration/non_partitioned` run:
```shell
uv run oellm-package-data --collection-dir tests/resources/integration/flat_release -m lint
```

See [docs/linter.md](docs/linter.md) for a quick start and how to read the log messages.


### The packager: oellm-package-data

The packager takes source files and applies decontamination, PII-masking, and sampling. The tool is idempotent,
if it fails, then run it again, and it will take of where it left.

To package the data in `tests/resources/integration/non_partitioned` run:
```shell
uv run oellm-package-data --collection-dir tests/resources/integration/flat_release
```
The program checks if output files exist, if they exist new data is not regenerated.
`oellm-package-data` can run in different modes, `--mode`. This controls what
part of the process will happen. Default is `release` mode packaging the final
result and output into `release-raw`. The mode `sample` do just a sampling according
to the `sample` section in `metadata.yaml`, and output into `sample`directory.

`--token-count` in `release` and `sample` mode counts tokens with the first tokenizer listed in `megatron-lm`.

### Merger
The merger reduces the number of files but still keeps semantics in paths, like language or quality.
The merger uses the `metadata.yaml` in provided collection-directory.
Input directory is the subdirectory `release-raw` and output to `release` subdirectory.
Metrics computed during `release` are aggregated.

The merger run after `oellm-package-data`.

To run local:
```shell
uv run oellm-package-data -m merge --collection-dir ${COLLECTION_DIR} --workers 1
```

### Tokenization
The mode `tokenize` turns files into Megatron-LM indexed datasets (`.bin` and `.idx`). It is controlled by the
optional `megatron-lm` section in `metadata.yaml`. Without the section the data cannot be tokenized, and the linter
logs a warning:

```yaml
megatron-lm:
  default:
    source: release
  openeurollm-256k:
    huggingface: openeurollm/tokenizer-256k
```

`default.source` is the directory under the collection directory to read files from. Every other entry is a
tokenizer, where `huggingface` is a HuggingFace Hub repo id or a local tokenizer path. The output of each tokenizer is
written to `megatron-lm/<entry name>`, so the example above writes to `megatron-lm/openeurollm-256k`. When several
tokenizers are listed, the files are tokenized with all of them. A tokenizer entry may set its own `source`.

Every document gets a beginning-of-document token prepended and an end-of-document token appended. They are set with
`bos` and `eos`, which default to `<bos>` and `<eos>`. Set a field to `false` to not add that token. Both fields can be
set in `default` or for each tokenizer, where the tokenizer value takes precedence:

```yaml
megatron-lm:
  default:
    source: release
    bos: false
  openeurollm-256k:
    huggingface: openeurollm/tokenizer-256k
  other-tokenizer:
    huggingface: org/other-tokenizer
    bos: <s>
    eos: </s>
```

```shell
uv run oellm-package-data --collection-dir <dir> --mode tokenize
```

### Propella Structure: oellm-propella-structure

This is only relevant if you want to annotate your data with propella annotations. It must run before packer.

The propella-structure tool filters source records based on IDs found in Propella parquet files. It reads
source files from `collection_dir/source`, looks up each record's ID in the propella directory, and
writes matching records to `collection_dir/propella-4b`. This is useful for structuring Propella data, according to
source data.

The Propella data is read into memory and requires about 50 times the size of the Propella parquet file. If
your Propella data does not fit into memory. Run ones per parquet file and then merge the result with
`oellm-propella-merge`. When pointing to an individual parquet file, the output directory will get an extra
level of directory named after the parquet file. This can be removed after `oellm-propella-merge`.

To run:
```shell
uv run oellm-propella-structure --collection-dir ${COLLECTION_DIR} --propella ${PROPELLA_DIR} --part ${part}
```

To run `oellm-propella-merge`:
```shell
uv run oellm-propella-merge --collection-dir ${COLLECTION_DIR} --part ${part}
```

### Tokenizer: oellm-tokenize

The tokenizer turns JSONL files into Megatron indexed datasets (`.bin` and `.idx`) without needing a collection
directory or a `metadata.yaml`. Every document is wrapped with the tokenizer's `<bos>` and `<eos>` tokens.

Arguments:
* `input` - a single file, or a directory where all `*.jsonl.zst` files below it are tokenized
* `-t`, `--tokenizer` - HuggingFace Hub repo id or local path to the tokenizer (required)
* `--text` - the field in each document holding the text, default `text`
* `-o`, `--output-dir` - where to write the output, default the current directory
* `-s`, `--slurm` - only tokenize the files assigned to the current SLURM array task, see below

For a directory, the relative path of each input file is kept under the output directory, and the file extension
is dropped, e.g. `input/deu_Latn/000.jsonl.zst` becomes `output/deu_Latn/000.bin` and `output/deu_Latn/000.idx`.
A metrics file is written next to each dataset. Files that already have output are skipped, so the tool can be
rerun after a failure.

To run:
```shell
uv run oellm-tokenize ${INPUT_DIR} --tokenizer ${TOKENIZER} --text text --output-dir ${OUTPUT_DIR}
```

To spread the files over a SLURM job array, add `--slurm`. Each array task tokenizes every
`SLURM_ARRAY_TASK_COUNT`-th file, starting at its `SLURM_ARRAY_TASK_ID`:
```shell
sbatch --array=0-9 --wrap "uv run oellm-tokenize ${INPUT_DIR} --tokenizer ${TOKENIZER} --output-dir ${OUTPUT_DIR} --slurm"
```

## Develop

Before checking in run tests, linting and formating:
```shell
uv run pre-commit
```

### Test megatron similarity

When working with tokenizer it is valuable to ensure compatibility with Megatron-LM.
These tests are skipped if megatron not installed. To install megatron do:

```shell
uv sync --extra megatron
```
