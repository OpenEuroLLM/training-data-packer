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

export COLLECTION_DIR=$1

time uv run oellm-package-merge --collection-dir ${COLLECTION_DIR} --workers 1 --slurm