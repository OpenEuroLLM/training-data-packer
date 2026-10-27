#!/bin/bash -x
#SBATCH --partition=standard
#SBATCH --job-name=tokenize-standalone
#SBATCH --output=logs/tokenize-standalone_%A_%a.out
#SBATCH --error=logs/tokenize-standalone_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

time singularity exec \
  --bind /appl/local/openeurollm \
  --bind /pfs/lustrep2/scratch \
  --bind /pfs/lustrep3/scratch \
  --bind /scratch \
  --bind /project \
  /scratch/project_465002530/containers/training-data-packer/training-data-packer.sif \
  oellm-tokenize --slurm $*
