#!/bin/bash
# Print a Markdown table with the status of the jobs scheduled by pipeline.sh and pipeline-auto-plan.sh.
# Each row is a collection dir, each column a step in the pipeline. A collection scheduled more than once only shows
# the latest jobs.
#
# 🏃🏻 running, ✅ completed, ❎ failed, 🕘 pending or waiting for a dependency, ❔ unknown to SLURM.
# An array job shows the worst state of its tasks: failed before running before pending before completed.
#
# Usage: ./pipeline-status.sh [jobs-csv]
# Example: ./pipeline-status.sh logs/jobs.csv

set -euo pipefail

JOBS_CSV=${1:-logs/jobs.csv}

if [ ! -f "${JOBS_CSV}" ]; then
  echo "No such file: ${JOBS_CSV}" >&2
  exit 1
fi

STATES_FILE=$(mktemp)
trap 'rm -f "${STATES_FILE}"' EXIT

JOB_IDS=$(tail -n +2 "${JOBS_CSV}" | cut -d, -f2- | tr ',' '\n' | grep -v '^$' | sort -u | paste -sd, - || true)
if [ -n "${JOB_IDS}" ]; then
  sacct --jobs="${JOB_IDS}" --allocations --noheader --parsable2 --format=JobID,State > "${STATES_FILE}"
fi

awk -F'|' '
  FILENAME != ARGV[1] && FNR == 1 {
    FS = ","
    $0 = $0
  }

  FILENAME == ARGV[1] {
    # Array tasks are reported as <job id>_<task> or <job id>_[<range>] while pending.
    split($1, id, "_")
    split($2, state, " ")
    if (state[1] ~ /^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED|REVOKED)$/) {
      failed[id[1]] = 1
    } else if (state[1] ~ /^(RUNNING|COMPLETING|CONFIGURING|STAGE_OUT|SIGNALING)$/) {
      running[id[1]] = 1
    } else if (state[1] == "COMPLETED") {
      completed[id[1]] = 1
    } else {
      pending[id[1]] = 1
    }
    next
  }

  function status(job_id) {
    if (job_id in failed) return "❎"
    if (job_id in running) return "🏃🏻"
    if (job_id in pending) return "🕘"
    if (job_id in completed) return "✅"
    return "❔"
  }

  FNR == 1 {
    columns = NF
    header = "| Collection |"
    separator = "|---|"
    for (i = 2; i <= NF; i++) {
      step = $i
      sub(/ job id$/, "", step)
      header = header " " step " |"
      separator = separator "---|"
    }
    next
  }

  NF > 0 {
    dir = $1
    sub(/\/+$/, "", dir)
    n = split(dir, parts, "/")
    name = parts[n]
    if (!(name in row)) order[++rows] = name
    row[name] = "| " name " |"
    for (i = 2; i <= columns; i++) row[name] = row[name] " " status($i) " |"
  }

  END {
    print header
    print separator
    for (i = 1; i <= rows; i++) print row[order[i]]
  }
' "${STATES_FILE}" "${JOBS_CSV}"
