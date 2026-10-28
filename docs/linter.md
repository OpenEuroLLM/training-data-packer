# Linter quick start

The linter checks a collection's `metadata.yaml` and directory layout before you spend cluster time on packaging.
Run it every time you create or change a `metadata.yaml`, and fix all errors before running `release`, `sample` or
`merge`.

## Run the linter

```bash
uv run oellm-package-data --collection-dir <collection-dir> --mode lint
```

`<collection-dir>` is the directory that holds `metadata.yaml`. The short form `-m lint` also works.

Exit codes:

| Exit code | Meaning                                                           |
|-----------|-------------------------------------------------------------------|
| `0`       | No errors. Warnings and info messages may still have been logged. |
| non-zero  | At least one error. The metadata must be fixed.                   |

The exit code makes it easy to use the linter in scripts:

```bash
uv run oellm-package-data --collection-dir "$DIR" --mode lint && sbatch --array=0-49 ./package.sh "$DIR"
```

## What it checks

- **Metadata file**: `metadata.yaml` exists and can be read.
- **Release parts**: every part in `release` follows the JSON schema, uses valid `sample` (`full`, `random`, `dynamic`,
  `wds+register`) and `pack` (`flat`, `tree`) values, and sets the fields each mode requires, for example `budget` for
  `random`, and `filter` and `parameters` for `dynamic`. For `dynamic`, the sampler function must be loadable.
- **Annotations**: every value in an `annotations` list is known and has a matching section in the metadata. A
  `propella-4b`, `nemo-curator` or `openai-privacy-filter` section that is defined must also be used.
- **Section references**: each `input` points to a section that exists.
- **Metadata matches directories**: a `uuid`, `sample`, `nugget` or annotation directory on disk must have a
  matching section in `metadata.yaml`.
- **Data files**: when a section directory exists, the linter reads the first record of each part and checks that
  required fields (such as `id` and `text`, or the `parallel` fields) are present. It also checks that part names
  match the subdirectories on disk.
- **Tokenizers**: the `megatron-lm` section is optional, and a warning is logged if it is missing. If it is set, it
  lists at least one tokenizer, every tokenizer can be loaded, its `bos` and `eos` tokens are single tokens in the
  tokenizer, and each `source` points to a section that exists. Directories under `megatron-lm/` that match no
  tokenizer give a warning.

The linter only reads a sample of the data, so a clean lint does not guarantee that every record is valid.

## Read the log messages

Each message has a level. Act on it as follows:

| Level     | What it means                                                         | What to do                         |
|-----------|-----------------------------------------------------------------------|------------------------------------|
| `ERROR`   | The metadata or data is invalid. The linter exits with non-zero code. | Must be fixed before packaging.    |
| `WARNING` | Something is missing or does not line up, often missing configuration.| Review carefully.                  |
| `INFO`    | Good practice recommendation.                                         | Consider applying it.              |

### ERROR: must be fixed

The linter stops at the first error and exits with a non-zero code. Fix it and run the linter again, because more
errors may be hidden behind the first one.

```text
ERROR | Metadata is invalid: In release.'shard01', `budget` is required when `sample` is `random`.
```

### WARNING: review carefully

Warnings do not change the exit code, but they often point at missing configuration or data. Check each one and make
sure you understand why it is there.

```text
WARNING | `propella-4b` data has not been created.
WARNING | Some parts have no directory in `source`: shard02
WARNING | In source, the expected fields propella-4b were not found in any of the sampled records.
```

A "data has not been created" warning is expected when an earlier pipeline step has not run yet. If that step
should already be done, the warning means data is missing or in the wrong place.

### INFO: good practice

Info messages are recommendations. They do not indicate a problem, but following them makes the metadata clearer
and the output more predictable, for example by setting a value explicitly instead of relying on a default.

```text
INFO | In release.'shard01', setting `prefix` is recommended when `pack` is `flat` to get unique names. The default value of `prefix` is `...`.
INFO | In release.'shard01', with sample mode 'full', the following fields have no effect and should not be set: budget.
```

The linter also logs the metadata checksum and content at `INFO` level at startup. These lines are for traceability
and need no action.

### Repeated messages are shown once

To keep the output short, each message is logged only once per code location. If the same problem affects several
parts, only the first one is shown. After fixing it, run the linter again to see whether other parts have the same
problem.
