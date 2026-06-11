# training-data-packer
Packaging annotated datasets into final training data. Purpose is to have a repeatable process and
well-packaged data to remove any data management in the training step.

The packer consists of the following tools:
* oellm-package-data - Take source files and apply decontamination, PII-masking, and sampling. Each file
    in the source directory gets a correspondent file with the data processed. The tool is idempotent, if it fails,
    then run it again, and it will take of where it left. This is the main tool.
* oellm-package-merge - This shall run after oellm-package-data and deduces the number of files to simplify
    tokenization and training.
* oellm-collect-metrics - Collect and summarize metrics from a collection directory.
* oellm-propella-structure - Structure Propella data based on source data structure. For each record in the
    source files, if its ID exists in the Propella data, it is written to the output. This arranges Propella
    data in the same oßrder as source data.
* oellm-propella-merge - If Propella data is to big to be in memory `oellm-propella-structure` can run on
  individual Propella-parquet files. Then use this tool to merge the results from all Propella-parquet files.

Both tools read a file `metadata.yaml` containing metadata about the structure and processing of the data.


## Related projects
This project uses data produced from several other projects:
* [Training data collection](https://github.com/OpenEuroLLM/training-data-collection) containing the
  metadata for OpenEuroLLM models
* [Decontamination](https://github.com/OpenEuroLLM/pretraining-decontamination)
* PII detection for [baby cycle](https://github.com/OpenEuroLLM/pii-masking-oellm)
  and [flag cucle](https://github.com/yann-ufal/pii-detection-solo)

After packaging is happening tokenization must happen before the training. This is
done according to the description [here](https://github.com/OpenEuroLLM/tokenizer/tree/main/baby-tokenization).

## Requirements
You must have `uv` installed, see [uv-homepage](https://docs.astral.sh/uv/).

Check out this [repo](https://github.com/OpenEuroLLM/training-data-packer#) and do:
```shell
uv sync --extra dev
uv run pre-commit install
```

Tip is to do an `uv sync` evry time pulling from the repository.

## Run

### The packager: oellm-package-data

The packager takes source files and applies decontamination, PII-masking, and sampling. Each file
in source directory gets a correspondent file in `output_dir` with processed data. The tool is idempotent,
if it fails, then run it again, and it will take of where it left.

To package the data in `tests/resources/integration/non_partitioned` run:
```shell
uv run oellm-package-data --input_dir tests/resources/integration/flat_release --output_dir tmp
```
The program checks if output files exist, if they exist new data is not regenerated.

It can also run via slurm:
```shell
sbatch --array=0-10 ./package.sh input-dir  output-dir
```

When using Slurm, data sharding is handled automatically across the task array.

Here is a full example running on Lumi:
```shell
sbatch --array=0-49 ./package.sh \
    /scratch/project_465002530/training/collection/baby/nemotron-cc-opus-1.1 \
    /scratch/project_465002530/training/collection/baby/nemotron-cc-opus-1.1/release_raw
```

### Merger: oellm-package-merge
The merger reduces the number of files but still keeps semantics in paths, like language or quality.
The merger uses the `metadata.yaml` in provided collection-directory. As input it use the subdirectory `release_raw` and
write the merged files to `release` subdirectory.

The merger run after `oellm-package-data`.

To run local:
```shell
uv run oellm-package-merge --collection-dir ${COLLECTION_DIR} --workers 1
```

It can also run via slurm:
```shell
sbatch --array=0-9 ./merge.sh \
    /scratch/project_465002530/training/collection/baby/nemotron-cc-opus-1.1
```

### Propella Structure: oellm-propella-structure

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

## Develop

Before checking in run tests, linting and formating:
```shell
uv run pre-commit
```
