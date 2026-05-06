# training-data-packer
Packaging annotated datasets into final training data. Purpose is to have a repeatable process and 
well packaged data to remove any data management in the training step.

## Run

To package the data in `tests/resources/integration/non_partitioned` run:
```shell
uv run oellm-package-data --input_dir tests/resources/integration/non_partitioned --output_dir tmp
```
The program checks if output files exist, if they exist new data is not regenerated. 

It can also run via slurm:
```shell
sbatch --array=0-10 ./package.sh input-dir  output-dir
```

When running with slurm it will automatically shard the data over the array tasks.



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
text: text
id: id
annotations:
- contamination:
  - owner: Department B
  - status: complete
- pii:
  - owner: Department A
  - status: complete
release:
  default:
    sample: wds+register
  eng_Latn:
    sample: random
    budget: 65%
  swe_Latn:
    sample: full

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