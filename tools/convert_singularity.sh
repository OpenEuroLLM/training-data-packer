#!/usr/bin/env bash

# This is aimed to run on Lumi. It pulls a OCI container from OpenEuroLLM
# container registry in GitHub.
# The pull convert the container into a SIF-container.

mkdir -p /tmp/$USER
export SINGULARITY_TMPDIR=/tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER
singularity pull docker://ghcr.io/openeurollm/training-data-packer:latest
