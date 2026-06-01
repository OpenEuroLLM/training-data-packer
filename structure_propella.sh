#!/bin/bash -x
#SBATCH --partition=small
#SBATCH --job-name=structure-propella
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/structure-propella_%A_%a.out
#SBATCH --error=logs/structure-propella_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=150G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

# Reading the propella data requires 2.2GB of memory per million rows. Make sure to adjust memory consumption.

export COLLECTION_DIR=$1
export PROPELLA_DIR=$2
if [ "$#" -eq 3 ]; then
  export PART="--part $3"
else
  export PART=""
fi
export LOGURU_LEVEL=INFO

time uv run oellm-propella-structure --collection-dir ${COLLECTION_DIR} --propella ${PROPELLA_DIR} $PART --slurm
