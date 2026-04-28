# training-data-packer
Packaging annotated datasets into final training data.

## Run

To package the data in `tests/resources/integration/non_partitioned` run:
```shell
uv run main.py --input_dir tests/resources/integration/non_partitioned --output_dir tmp
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
text: text
id: id
wds: true
registers: true
annotations:
- contamination:
  - owner: Prompsit
  - status: complete
- pii:
  - owner: AI Sweden
  - status: complete
```

Currently, the packager specifically requires `id` and `text` fields. If your source
data uses different names for these attributes, use these parameters to map your
existing fields to the required document ID and content fields.