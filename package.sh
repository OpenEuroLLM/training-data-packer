#!/bin/bash -x
#SBATCH --partition=small
#SBATCH --job-name=package
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/package_%A_%a.out
#SBATCH --error=logs/package_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

export COLLECTION_DIR=$1
if [ "$#" -eq 2 ]; then
  export MODE="--mode $2"
else
  export MODE=""
fi
export LOGURU_LEVEL=INFO

time uv run oellm-package-data --collection-dir ${COLLECTION_DIR} $MODE --workers 8 --slurm
