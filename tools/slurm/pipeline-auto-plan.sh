#!/bin/bash
# Submit a package job (default release mode), a merge job, and a tokenizer job.
# A task only starts if previous task succeeds.
# The array size and time of each job are calculated from the files in the collection, see `--mode plan`.
#
# Usage: ./pipeline-auto-plan.sh <collection-dir>
# Example: ./pipeline-auto-plan.sh /scratch/project_465002530/training/collection/flag/finepdfs-edu-1.0.0

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <collection-dir>" >&2
  exit 1
fi

ACCOUNT=project_465002530
PACKER=/scratch/project_465002530/containers/training-data-packer/oellm-package-data.sh

COLLECTION_DIR=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export LOGURU_LEVEL=${LOGURU_LEVEL:-INFO}

mkdir -p logs
if [ ! -f logs/jobs.csv ]; then
    echo Collection dir,Package job id,Merge job id,Tokenize job id > logs/jobs.csv
fi

echo Run linter before scheduling pipeline
${PACKER} --collection-dir ${COLLECTION_DIR} --mode lint

echo Calculate array size and time for each job
PLAN=$(${PACKER} --collection-dir ${COLLECTION_DIR} --mode plan)
echo "${PLAN}"
PACKAGE_ARGS=$(awk '$1 == "package" { $1 = ""; print }' <<< "${PLAN}")
MERGE_ARGS=$(awk '$1 == "merge" { $1 = ""; print }' <<< "${PLAN}")
TOKENIZE_ARGS=$(awk '$1 == "tokenize" { $1 = ""; print }' <<< "${PLAN}")

PACKAGE_JOB_ID=$(sbatch --account=${ACCOUNT} --parsable ${PACKAGE_ARGS} "${SCRIPT_DIR}/package.sh" "${COLLECTION_DIR}" | cut -d';' -f1)
echo "Submitted package job ${PACKAGE_JOB_ID}"

MERGE_JOB_ID=$(sbatch --account=${ACCOUNT} --parsable ${MERGE_ARGS} --dependency="afterok:${PACKAGE_JOB_ID}" --kill-on-invalid-dep=yes \
  "${SCRIPT_DIR}/merge.sh" "${COLLECTION_DIR}" | cut -d';' -f1)
echo "Submitted merge job ${MERGE_JOB_ID}, runs after package job ${PACKAGE_JOB_ID} succeeds"

TOKENIZE_JOB_ID=$(sbatch --account=${ACCOUNT} --parsable ${TOKENIZE_ARGS} --dependency="afterok:${MERGE_JOB_ID}" --kill-on-invalid-dep=yes \
  "${SCRIPT_DIR}/tokenize.sh" "${COLLECTION_DIR}" | cut -d';' -f1)
echo "Submitted tokenize job ${TOKENIZE_JOB_ID}, runs after merge job ${MERGE_JOB_ID} succeeds"

echo ${COLLECTION_DIR},${PACKAGE_JOB_ID},${MERGE_JOB_ID},${TOKENIZE_JOB_ID} >> logs/jobs.csv
