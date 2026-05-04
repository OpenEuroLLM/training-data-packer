#!/bin/bash -x
#SBATCH --partition=debug
#SBATCH --job-name=package
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/package_%A_%a.out
#SBATCH --error=logs/package_%A_%a.out
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --account=project_465002530

export IN_DIR=/scratch/project_465002530/training/collection/baby/dclm-1.0
export OUT_DIR=/scratch/project_465002530/users/magnrune/tmp/

uv run main.py --input_dir ${IN_DIR} --output_dir ${OUT_DIR} --workers 1 --slurm