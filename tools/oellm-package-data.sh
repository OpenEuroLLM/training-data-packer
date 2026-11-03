#!/usr/bin/env bash

# This script is intended for the OpenEuroLLM project on Lumi.
# Wrapper script for the Singularity container to ensure needed directories are bound.
# Set TRAINING_DATA_PACKER_SIF to use an alternative SIF file.

SIF=${TRAINING_DATA_PACKER_SIF:-/scratch/project_465002530/containers/training-data-packer/training-data-packer.sif}

singularity exec \
  --bind /appl/local/openeurollm \
  --bind /pfs/lustrep2/scratch \
  --bind /pfs/lustrep3/scratch \
  --bind /scratch \
  --bind /project \
  "${SIF}" \
  oellm-package-data $*
