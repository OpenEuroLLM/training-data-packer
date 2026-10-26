#!/bin/bash -x
#SBATCH --partition=small
#SBATCH --job-name=merge
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/merge_%A_%a.out
#SBATCH --error=logs/merge_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

# To calculate array size and time:
# A single node process 2-3GB/minute zstd compressed json.
# If tree array size shall not be larger than number of parts. If flat array is only one task.

export COLLECTION_DIR=$1
export LOGURU_LEVEL=INFO

time /scratch/project_465002530/containers/training-data-packer/oellm-package-data.sh --mode merge --collection-dir ${COLLECTION_DIR} --workers 1
