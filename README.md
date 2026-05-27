# training-data-packer
Packaging annotated datasets into final training data. Purpose is to have a repeatable process and
well packaged data to remove any data management in the training step.

The packer consist of four tools:
* oellm-package-data - Take source files and apply decontamination, PII-masking and sampling. Each file in
    in source get a correspondent file with the data processed. The tool is idempotent, if it fails then
    run it again and it will take of where it left.
* oellm-package-merge - This shall run after oellm-package-data and deduces the number of files to simplify
    tokenization and training.
* oellm-collect-metrics - Collect and summarize metrics from a collection directory.
* oellm-propella-structure - Structure Propella data based on source data structure. For each record in the
    source files, if its ID exists in the propella data, it is written to the output. This arrange propella
    data in same order as source data.

Both tools read a file `metadata.yaml` containing metadata about the structure and processing of the data.

## Requirements
You must have `uv` installed, see [uv-homepage](https://docs.astral.sh/uv/).

Check out this [repo](https://github.com/OpenEuroLLM/training-data-packer#) and do:
```shell
uv sync --extra dev
uv run pre-commit install
```

Tip is to do an `uv sync` evry time pulling from the repository.

## Run

### Packager

The packager takes source files and apply decontamination, PII-masking and sampling. Each file in
in source gets a correspondent file in `output_dir` with processed data. The tool is idempotent,
if it fails then run it again and it will take of where it left.

To package the data in `tests/resources/integration/non_partitioned` run:
```shell
uv run oellm-package-data --input_dir tests/resources/integration/flat_release --output_dir tmp
```
The program checks if output files exist, if they exist new data is not regenerated.

It can also run via slurm:
```shell
sbatch --array=0-10 ./package.sh input-dir  output-dir
```

When utilizing Slurm, data sharding is handled automatically across the task array.

Here is a full example running on Lumi:
```shell
sbatch --array=0-49 ./package.sh \
    /scratch/project_465002530/training/collection/baby/nemotron-cc-opus-1.1
    /scratch/project_465002530/training/collection/baby/nemotron-cc-opus-1.1/release_raw
```

### Merger
The merger exist to reduce the number of files but still keep semantics in paths, like language or quality.
The merger uses the `metadata.yaml` in provided collection-directory. As input it use the subdirectory `release_raw` and
write the merged files to `release` sub-directory.

To run local:
```shell
uv run oellm-package-merge --collection-dir ${COLLECTION_DIR} --workers 1
```

It can also run via slurm:
```shell
sbatch --array=0-9 ./merge.sh \
    /scratch/project_465002530/training/collection/baby/nemotron-cc-opus-1.1
```

### Propella Structure

The propella-structure tool filters source records based on IDs found in propella parquet files. It reads
source files from `collection_dir/source`, looks up each record's ID in the propella directory, and
writes matching records to `collection_dir/propella`. This is useful for structuring propella data according to
source data.

To run:
```shell
uv run oellm-propella-structure --collection-dir ${COLLECTION_DIR} --propella ${PROPELLA_DIR}
```

## Develop

Before checking in run tests, linting and formating:
```shell
uv run pre-commit
```
