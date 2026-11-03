#!/bin/bash -x
#SBATCH --partition=standard
#SBATCH --job-name=tokenize-standalone
#SBATCH --output=logs/tokenize-standalone_%A_%a.out
#SBATCH --error=logs/tokenize-standalone_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

export HF_HOME=/scratch/project_465002530/cache/huggingface

# Set TRAINING_DATA_PACKER_SIF to use an alternative SIF file.
SIF=${TRAINING_DATA_PACKER_SIF:-/scratch/project_465002530/containers/training-data-packer/training-data-packer.sif}

time singularity exec \
  --bind /appl/local/openeurollm \
  --bind /pfs/lustrep2/scratch \
  --bind /pfs/lustrep3/scratch \
  --bind /scratch \
  --bind /project \
  "${SIF}" \
  oellm-tokenize-standalone --slurm $*
