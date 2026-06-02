# Pipeline release Flag

This document provides an overview of data processing pipeline of the training
data packager.

The pipeline consists of the following main:
* Propella structure - Pre-processing of Propella data to match source data. This produce the `propella-4b` directory.
* Propella merge - Merge Propella data if not fit into memory.
* PII Detection - Externa step to detect PII data. This produce the `pii` directory.
* Contamination detection - External step to detect data existing in benchmarks.
  This produce the `contamination` directory.
* package - Filter and format the data. This produce data to a `release_raw` directory.
* merge - Merge into reduced number of shards with even size. This produce data to `release` directory.
* Tokenization - External step to tokenize the data. This produce the `megatron-lm` directory.

```mermaid
---
title: "Packaging pipeline"
---
graph TD;
    Source[(source)]-->Structure["Structure propella"]
    subgraph Propella Structure
        PropellaSource[(Propella source data)]-->Structure
        Structure-->Propella4B[("Propella-4b")]
    end
    subgraph Propella Merge
        Propella4B[("propella-4b")]-.->PropellaMerge[Merge Propella files]
        PropellaMerge-.->Propella4B
    end
    Source-->PIIDetection
    Source-->ContaminationDetection
    PIIDetection[[PII Detection]]-->PIIdata
    PIIdata[(pii)]-->PII
    ContaminationDetection[[Contamination Detection]]-->ContaminationData
    ContaminationData[(contamination)]-->Contamination
    Source-->Align["Align field names"]
    subgraph Package data
        Align-->Scrub["Scrub unwanted fields"]
        Scrub-->PropellaAnotation[[Propella annotation]]
        Propella4B-->PropellaAnotation
        PropellaAnotation-->PII[PII masking]
        PII-->Contamination["Decontamination"]
        Contamination-->Block["Block list filter"]
        Block-->Sampler["Sample data to budget"]
        Sampler-->Writer["Write to disk (.jsonl.zst)"]
        Writer-->ReleaseRaw[(release_raw)]
    end
    subgraph Package merge
        ReleaseRaw-->Merger([Merge into shards])
        Merger-->ShardWriter([Write merged shards])
    end
    ShardWriter-->Release[(release)]
    Release-->Tokenization[[Tokenization]]
```
