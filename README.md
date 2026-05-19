# training-data-packer
Packaging annotated datasets into final training data. Purpose is to have a repeatable process and 
well packaged data to remove any data management in the training step.

The packer consist of two tools:
* oellm-package-data - Take source files and apply decontamination, PII-masking and sampling. Each file in
    in source get a correspondent file with the data processed. The tool is idempotent, if it fails then 
    run it again and it will take of where it left.
* oellm-package-merge - This shall run after oellm-package-data and deduces the number of files to simplify
    tokenization and training.

Both tools read a file `metadata.yaml` containing metadata about the structure and processing of the data.

## Requirements
You must have `uv` installed, see [uv-homepage](https://docs.astral.sh/uv/).

Check out this [repo](https://github.com/OpenEuroLLM/training-data-packer#) and do `uv sync`.

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

## Develop

Before checking in run tests, linting and formating:
```shell
uv run --with pytest pytest
uv run ruff check --fix 
uv run ruff format
```

## Expected structure
```
input_dir/
    metadata.yaml
    source/
    pii/
    contamination/
```
Source is where the source data is. If it contains a file `part_5/file_3.jsonl.zst`
it is expected that the directories `pii` and `contamination` both has a file with
the same name.

The file `metadata.yaml` contains metadata to help package the data. Example:
```yaml
name: HPLT 3.0
url: https://hplt-project.org/datasets/v3.0
suffix: .jsonl.gz
text: text
id: id
annotations:
  contamination:
    owner: Department B
    status: complete
  pii:
    owner: Department A
    status: complete
    suffix: .jsonl
release:
  default:
    sample: wds+register
  eng_Latn:
    sample: random
    budget: 65%
  swe_Latn:
    sample: full
    block:
    - 6d1f3087-fcdb-4a84-bd64-00edc2862472
```

In the example above we tell tat field containing the text to be masked and output is in field
`text`, which is the default. Document id are stored in `id`. If `id` is in a sub-record of the
record use a dot-notation, `metadata.WARC-Record-ID`. 

The `release` section contains definitions for different sub-dataset of our dataset. In this case
it is one per language. The `default` is used if there are no sub-dataset or if it is not explicitly
pointed out. 
In a release a sampling method can be pointed out in the field `sample`. Three methods are currently
supported:

* `full` - Keep all data
* `random` - Keep a random fraction of the dataset based on the field `budget`. `65%` means we keep 65% of the records.
* `wds+register` - Requires the documents are decorated with Web Docs Scorer. Down- and up-sampling is done based on the scores.

Each release can have a block-list, field `block`, containing a list of document to be removed. The 
intended use is to remove documents with issues, these shall be kept to a minimum.