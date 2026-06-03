#!/bin/bash -x
#SBATCH --partition=small
#SBATCH --job-name=propella-merge
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/propella-merge_%A_%a.out
#SBATCH --error=logs/propella-merge_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

export COLLECTION_DIR=$1
export PART=$2

export LOGURU_LEVEL=INFO

time uv run oellm-propella-merge --collection-dir ${COLLECTION_DIR} --part $PART --workers 8 --slurm
