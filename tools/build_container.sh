#!/usr/bin/env bash

# This script build container for both arm64 and amd64 on Mac using
# [Apple Container](https://github.com/apple/container). Container is pushed to OpenEuroLLM
# container registry in GitHub.

container build --arch arm64 --arch amd64 -t ghcr.io/openeurollm/training-data-packer:latest
container image push ghcr.io/openeurollm/training-data-packer:latest
