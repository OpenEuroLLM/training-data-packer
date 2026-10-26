#!/bin/bash
# Submit a package job (default release mode), a merge job, and a tokenizer job.
# A task only starts if previous task succeeds.
#
# Usage: ./pipeline.sh <collection-dir> <package-array> <merge-array> <tokenize-array>
# Example: ./pipeline.sh /scratch/project_465002530/training/collection/flag/finepdfs-edu-1.0.0 0-4 0-9 0-1

set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <collection-dir> <package-array> <merge-array> <tokenize-array>" >&2
  exit 1
fi

ACCOUNT=project_465002530
SCRIPT_DIR=./tools/slurm

COLLECTION_DIR=$1
PACKAGE_ARRAY=$2
MERGE_ARRAY=$3
TOKENIZE_ARRAY=$4
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

mkdir -p logs
if [ ! -f /tmp/foo.txt ]; then
    echo Collection dir,Package job id,Merge job id,Tokenize job id > logs/jobs.csv
fi

echo Run linter before scheduling pipeline
/scratch/project_465002530/containers/training-data-packer/oellm-package-data.sh --collection-dir ${COLLECTION_DIR} --mode lint

PACKAGE_JOB_ID=$(sbatch --account=${ACCOUNT} --parsable --array="${PACKAGE_ARRAY}" "${SCRIPT_DIR}/package.sh" "${COLLECTION_DIR}" | cut -d';' -f1)
echo "Submitted package job ${PACKAGE_JOB_ID}"

MERGE_JOB_ID=$(sbatch --account=${ACCOUNT} --parsable --array="${MERGE_ARRAY}" --dependency="afterok:${PACKAGE_JOB_ID}" --kill-on-invalid-dep=yes \
  "${SCRIPT_DIR}/merge.sh" "${COLLECTION_DIR}" | cut -d';' -f1)
echo "Submitted merge job ${MERGE_JOB_ID}, runs after package job ${PACKAGE_JOB_ID} succeeds"

TOKENIZE_JOB_ID=$(sbatch --account=${ACCOUNT} --parsable --array="${TOKENIZE_ARRAY}" --dependency="afterok:${MERGE_JOB_ID}" --kill-on-invalid-dep=yes \
  "${SCRIPT_DIR}/tokenize.sh" "${COLLECTION_DIR}" | cut -d';' -f1)
echo "Submitted merge job ${TOKENIZE_JOB_ID}, runs after package job ${MERGE_JOB_ID} succeeds"

echo ${COLLECTION_DIR},${PACKAGE_JOB_ID},${MERGE_JOB_ID},${TOKENIZE_JOB_ID} >> logs/jobs.csv
