#!/usr/bin/env bash

# This is aimed to run on Lumi. It pulls a OCI container from OpenEuroLLM
# container registry in GitHub.
# The pull convert the container into a SIF-container.

TAG=$1

mkdir -p /tmp/$USER
export SINGULARITY_TMPDIR=/tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

singularity pull training-data-packer-${TAG}.sif docker://ghcr.io/openeurollm/training-data-packer:${TAG}
