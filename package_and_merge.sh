#!/bin/bash
# Submit a package job (default release mode) and a merge job that starts only if every package task succeeds.
#
# Usage: ./package_and_merge.sh <collection-dir> <package-array> <merge-array>
# Example: ./package_and_merge.sh /scratch/project_465002530/training/collection/flag/finepdfs-edu-1.0.0 0-4 0-9

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <collection-dir> <package-array> <merge-array>" >&2
  exit 1
fi

COLLECTION_DIR=$1
PACKAGE_ARRAY=$2
MERGE_ARRAY=$3
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

mkdir -p logs

PACKAGE_JOB_ID=$(sbatch --parsable --array="${PACKAGE_ARRAY}" "${SCRIPT_DIR}/package.sh" "${COLLECTION_DIR}" | cut -d';' -f1)
echo "Submitted package job ${PACKAGE_JOB_ID}"

MERGE_JOB_ID=$(sbatch --parsable --array="${MERGE_ARRAY}" --dependency="afterok:${PACKAGE_JOB_ID}" --kill-on-invalid-dep=yes \
  "${SCRIPT_DIR}/merge.sh" "${COLLECTION_DIR}" | cut -d';' -f1)
echo "Submitted merge job ${MERGE_JOB_ID}, runs after package job ${PACKAGE_JOB_ID} succeeds"
