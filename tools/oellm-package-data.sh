#!/usr/bin/env bash

# This script is intended for the OpenEuroLLM project on Lumi.
# Wrapper script for the Singularity container to ensure needed directories are bound.

singularity exec \
  --bind /appl/local/openeurollm \
  --bind /pfs/lustrep2/scratch \
  --bind /pfs/lustrep3/scratch \
  --bind /scratch \
  --bind /project \
  /scratch/project_465002530/containers/training-data-packer/training-data-packer.sif \
  oellm-package-data $*
