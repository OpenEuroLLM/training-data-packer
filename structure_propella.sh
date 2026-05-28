#!/bin/bash -x
#SBATCH --partition=small
#SBATCH --job-name=structure-propella
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/structure-propella_%A_%a.out
#SBATCH --error=logs/structure-propella_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

export COLLECTION_DIR=$1
export PROPELLA_DIR=$2
export LOGURU_LEVEL=INFO

time uv run oellm-propella-structure --collection-dir ${COLLECTION_DIR} --propella ${PROPELLA_DIR} --workers 8 --slurm
