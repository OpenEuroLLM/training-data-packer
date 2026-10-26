#!/bin/bash -x
#SBATCH --partition=standard
#SBATCH --job-name=tokenize
#SBATCH --output=logs/tokenize_%A_%a.out
#SBATCH --error=logs/tokenize_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

# To calculate array size and time:
# A single node process 2-3GB/minute zstd compressed json. So use 2GB/minute as value.
# A single file cannot be split over several nodes.

export COLLECTION_DIR=$1
export LOGURU_LEVEL=INFO

time /scratch/project_465002530/containers/training-data-packer/oellm-package-data.sh --collection-dir ${COLLECTION_DIR} --mode tokenize
