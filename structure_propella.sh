#!/bin/bash -x
#SBATCH --partition=small
#SBATCH --job-name=structure-propella
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/structure-propella_%A_%a.out
#SBATCH --error=logs/structure-propella_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

export COLLECTION_DIR=$1
export PROPELLA_DIR=$2
export LOGURU_LEVEL=INFO

time uv run oellm-structure-propella --colection-dir ${COLLECTION_DIR} --propella ${PROPELLA_DIR} --propella --workers 8 --slurm
