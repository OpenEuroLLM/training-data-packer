# Training data packager within OpenEuroLLM

The training data packager is developed within the
[OpenEuroLLM](https://https://openeurollm.eu/) project.
The task for development is Task 3.5 Training Data Collection.
The task is dependent upon many other tasks within the project. The following
figure illustrates the position of Task 3.5 within the OpenEuroLLM project.

```mermaid
---
title: "Task 3.5 Position within the OpenEuroLLM project"
---
graph TD;
    T3.5["`T3.5 Training Data Configuration & Packaging
    Clean and unify data
    Mask PII
    Remove contaminated data (Decontamination)
    Sample data according to composition
    Tokenization
    `"]
    T3.1["T3.1 Training Data Acquisition and Analysis"]-->T3.3
    T3.1-->T3.4
    T3.1-- Data -->T3.5
    T3.2["T3.2 Targeted Training Data Sourcing"]-->T3.3
    T3.2-->T3.4
    T3.2-- Data -->T3.5
    T3.3["T3.3 Training Data Enrichment"]-- Contamination data -->T3.5
    T3.4["T3.4 Training Data Regulatory Compliance"]-- PII data -->T3.5

    T4.3["T4.3 Task Dataset composition and processing"]-- Budgets -->T3.5
    T3.5-- Tokenized data -->T4.4["T4.4 Large-scale foundation model training"]
    T3.5-->D3.2([D3.2 Initial dataset release])
    T3.5-->D3.3([D3.3 Final dataset release])
```
