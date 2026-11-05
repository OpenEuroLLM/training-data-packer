#!/bin/bash
# Print a Markdown table, aligned so it is readable as plain text too, with the status of the jobs scheduled by
# pipeline.sh and pipeline-auto-plan.sh. Each row is a collection dir, sorted by name, each column a step in the
# pipeline. A collection scheduled more than once only shows the latest jobs.
#
# 🏃🏻 running, ✅ completed, ❌ failed, 🕘 pending or waiting for a dependency, ❔ unknown to SLURM.
# An array job shows the worst state of its tasks: failed before running before pending before completed.
# A completed job also shows its run time as [D-]HH:MM:SS, from the start of its first task to the end of its last.
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
  # Print Start and End as seconds since the epoch, so the run time is easy to compute.
  SLURM_TIME_FORMAT=%s sacct --jobs="${JOB_IDS}" --allocations --noheader --parsable2 \
    --format=JobID,State,Start,End > "${STATES_FILE}"
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
    # Start and End are "Unknown" or "None" until the task has started or ended.
    if ($3 ~ /^[0-9]+$/ && (!(id[1] in start) || $3 < start[id[1]])) start[id[1]] = $3
    if ($4 ~ /^[0-9]+$/ && (!(id[1] in end) || $4 > end[id[1]])) end[id[1]] = $4
    next
  }

  function run_time(job_id,    seconds, days) {
    if (!(job_id in start) || !(job_id in end)) return ""
    seconds = end[job_id] - start[job_id]
    days = int(seconds / 86400)
    seconds -= days * 86400
    return sprintf(" %s%02d:%02d:%02d", days > 0 ? days "-" : "", seconds / 3600, seconds % 3600 / 60, seconds % 60)
  }

  function status(job_id) {
    if (job_id in failed) return "❌"
    if (job_id in running) return "🏃🏻"
    if (job_id in pending) return "🕘"
    if (job_id in completed) return "✅"
    return "❔"
  }

  # Store a cell with its display width. length() counts bytes or characters depending on the awk and locale, so
  # it is only used on ASCII text and the width of an emoji, two columns in a terminal, is passed in by the caller.
  function set_cell(r, c, text, width) {
    cell[r, c] = text
    cell_width[r, c] = width
    if (width > max_width[c]) max_width[c] = width
  }

  function repeat(char, n,    s) {
    s = ""
    while (n-- > 0) s = s char
    return s
  }

  function print_row(r,    c, line) {
    line = "|"
    for (c = 1; c <= columns; c++) line = line " " cell[r, c] repeat(" ", max_width[c] - cell_width[r, c]) " |"
    print line
  }

  FNR == 1 {
    columns = NF
    set_cell(0, 1, "Collection", length("Collection"))
    for (i = 2; i <= NF; i++) {
      step = $i
      sub(/ job id$/, "", step)
      set_cell(0, i, step, length(step))
    }
    next
  }

  NF > 0 {
    dir = $1
    sub(/\/+$/, "", dir)
    n = split(dir, parts, "/")
    name = parts[n]
    if (!(name in row)) {
      row[name] = ++rows
      names[rows] = name
    }
    r = row[name]
    set_cell(r, 1, name, length(name))
    for (i = 2; i <= columns; i++) {
      icon = status($i)
      time = icon == "✅" ? run_time($i) : ""
      set_cell(r, i, icon time, 2 + length(time))
    }
  }

  END {
    print_row(0)
    separator = "|"
    for (c = 1; c <= columns; c++) separator = separator repeat("-", max_width[c] + 2) "|"
    print separator
    # Insertion sort, since asort() is gawk only.
    for (i = 2; i <= rows; i++) {
      name = names[i]
      for (j = i - 1; j >= 1 && names[j] > name; j--) names[j + 1] = names[j]
      names[j + 1] = name
    }
    for (i = 1; i <= rows; i++) print_row(row[names[i]])
  }
' "${STATES_FILE}" "${JOBS_CSV}"
