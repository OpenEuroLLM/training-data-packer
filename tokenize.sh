#!/bin/bash -x
#SBATCH --partition=standard
#SBATCH --job-name=tokenize
#SBATCH --output=logs/tokenize_%A_%a.out
#SBATCH --error=logs/tokenize_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

export COLLECTION_DIR=$1
export LOGURU_LEVEL=INFO

time uv run oellm-package-data --collection-dir ${COLLECTION_DIR} --mode tokenize
