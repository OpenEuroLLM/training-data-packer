# Metrics

For each file processed(`filname`) a metrics file is produce named `.filename.metrics.json`.

The metric file looks like:

```json
{
  "input": {
    "document_read": 75094
  },
  "pii_masker": {
    "document_documents": 1731,
    "pii_documents": 124204
  },
  "contamination": {
    "document_removed": 13,
    "blocklist_length": 4377
  },
  "block_list": {
    "document_removed": 1,
    "blocklist_length": 1
  },
  "output": {
    "document_written": 75081,
    "size_bytes": 8402134
  }
}
```

## Collecting metrics
To collect and summarize metrics for an entire collection, use the `oellm-collect-metrics` tool:

```shell
uv run oellm-collect-metrics --collection-dir ${COLLECTION_DIR}
```

This will read all `.filename.metrics.json` files in the `release_raw` directory and its subdirectories,
sum up all numeric values, and write a summary to `metrics.json` in the `release_raw` directory.

If `metadata.yaml` has `release.default.pack` set to `tree`, the tool will instead create a `metrics.json`
file for each release section (except `default`) in their respective subdirectories under `release_raw`.

## Metric description

### Input
* `document_read` - Number of documents read from file


### Count
* `documents` - Number of documents, if last in chain it shall be same as output.
* `segments` - Number of document segments, correspond to number of new lines.
* `tokens` - Number of tokens, tokenizer used described in `tokenizer` field.
* `tokenizer` - Name of tokenizer used.
* `characters` - Number of characters.
* `unique_keys` - Number of unique document keys. Ideally same as documents.


### Dynamic sampler
* `document_read` - Documents read into the sampler.
* `document_written` - Number document output by the sampler.
* `document_removed` - Documents removed by sampler.
* `document_upsampled` - Document upsampled by sampler.
* `sampler_ratio_exceptions` - Number of exceptions from sampler. Ideally 0.

### Filter
This metric record is the same for contamination and blocklist.
* `document_removed` - Number of documents removed.
* `blocklist_length` - Number of documents in the filter list. If everything align, it shall be equal to
`document_removed`.

If `document_removed` greater than `blocklist_length` it is a sign of key duplication.
If `document_removed` are less than `blocklist_length` it is an indication of indata is not all documents.

### Parallel synthetic ID
Metric of synthetic id has been generated for parallel languages.
* `document_processed` - Number of document processed.

### Parallel merger
* `document_processed` - Language pair processed.
* `document_written` - Document written from parallel merger.

Quote between `document_processed` and `document_written` shall be close to `parallel.count` in metadata.
It will not be exact since number of documents may not be even divided by count.

### PII-masking
* `document_masked` - Number of documents PII was masked.
* `pii_records` - Number of unique document id in PII data.

If document ids are unique and PII records are stored by part, `document_masked` and `pii_records` will be the same.

### Propella
* `document_processed` - Documents processed with propella data.
* `document_unmatched` - Number of documents with no matching propella data.

### Output
* `document_written` - Number of documents written to output file.
