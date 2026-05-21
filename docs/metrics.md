# Metrics

For each file processed(`filname`) a metrics file is produce named `.filename.metrics.json`.

The metric file looks like:

```json
{
  "input": {
    "lines_read": 75094
  },
  "pii_masker": {
    "masked_documents": 1731,
    "pii_documents": 124204
  },
  "contamination": {
    "removed": 13,
    "list_length": 4377
  },
  "block_list": {
    "list_length": 1,
    "removed": 1
  },
  "output": {
    "lines_written": 75081
  }
}
```

## Collecting metrics
To collect and summarize metrics for an entire collection, use the `oellm-collect-metrics` tool:

```shell
uv run oellm-collect-metrics --collection-dir ${COLLECTION_DIR}
```

This will read all `.filename.metrics.json` files in the collection directory and its subdirectories, 
sum up all numeric values, and write a summary to `metrics.json` in the root of the collection directory.

## Metric description

### input.lines_read
Number of lines read from input file.

### pii_masker.masked_documents
Number of documents where PII was masked.

### pii_masker.pii_documents
Number of unique documents in pii data. If everything match it shall be equal to
`pii_masker.masked_documents`.

### contamination.removed
Number of documents removed due to contamination. 

### contamination.list_length
Number of documents in contamination data. If everything match it shall be equal to
`contamination.removed`.

### block_list.removed
Number of documents removed due to blocklist. 

### block_list.list_length
Length of block list in `metadata.yaml`. If everything match it shall be equal to
`block_list.removed`.

### output.lines_written
Number of lines written in output.
