#!/usr/bin/env bash
#SBATCH --job-name=reddit_monitor
#SBATCH --output=logs/reddit_monitor.%j.out
#SBATCH --error=logs/reddit_monitor.%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

###############################################################################
#  Array-job monitor: cancels + requeues tasks whose log files are silent
#  longer than --timeout minutes.
###############################################################################

# ----------------------------  Default settings  ----------------------------
JOB_NAME="reddit_pipeline"   # original name (we’ll truncate internally)
JOB_ID=""                    # allow passing root job-id directly
LOG_DIR="logs"
TIMEOUT_MINUTES=120
CHECK_INTERVAL=300           # seconds   (5 min)
DRY_RUN=false

# ---------------------------  Parse CLI arguments  --------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --job-name)   JOB_NAME="$2";      shift 2 ;;
        --job-id)     JOB_ID="$2";        shift 2 ;;
        --log-dir)    LOG_DIR="$2";       shift 2 ;;
        --timeout)    TIMEOUT_MINUTES="$2"; shift 2 ;;
        --interval)   CHECK_INTERVAL="$2";  shift 2 ;;
        --dry-run)    DRY_RUN=true;       shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [OPTIONS]

  --job-name NAME     Root job name to monitor (default: reddit_pipeline)
  --job-id   ID       Root job ID; bypass name lookup
  --log-dir  DIR      Log directory (default: logs)
  --timeout  MINUTES  Inactivity timeout per task (default: 30)
  --interval SECONDS  Poll interval (default: 300)
  --dry-run           Print actions without scancel/scontrol
  --help              Show this help
EOF
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "----------  SLURM array monitor  ----------"
echo "Job name  : $JOB_NAME"
echo "Job ID    : ${JOB_ID:-'(auto-detect)'}"
echo "Logs      : $LOG_DIR"
echo "Timeout   : $TIMEOUT_MINUTES min"
echo "Interval  : $CHECK_INTERVAL sec"
echo "Dry-run   : $DRY_RUN"
echo "-------------------------------------------"

# ----------------------------  Helper functions  ----------------------------
get_job_id() {
    # Return root job-id for the given JOB_NAME (respects SLURM’s 10-char limit)
    local name="$1"
    local short_name="${name:0:10}"

    # Exact match on the truncated name
    local id
    id=$(squeue --name="$short_name" --noheader --format="%A" | head -n1)
    [[ -n $id ]] && { echo "$id"; return; }

    # Fallback: search the (id,jobname) list
    id=$(squeue --noheader --format="%A %j" | awk -v n="$short_name" '$2==n{print $1;exit}')
    echo "$id"
}

is_task_running() { squeue --job="${1}_${2}" --noheader --format="%T" | grep -q RUNNING; }

is_job_running() { squeue --job="$1" --noheader --format="%T" | grep -qE '(PENDING|RUNNING|COMPLETING)'; }

file_mtime()     { [[ -f $1 ]] && stat -c %Y "$1" 2>/dev/null || echo 0; }

cancel_and_requeue() {
    local jid="$1" tid="$2" full="${jid}_${tid}"
    $DRY_RUN && { echo "[DRY-RUN] scancel $full && scontrol requeue $full"; return; }
    echo "Cancelling $full …"; scancel "$full" && sleep 2
    echo "Requeueing  $full …"; scontrol requeue "$full"
}

check_tasks() {
    local jid="$1" now timeout_sec=$((TIMEOUT_MINUTES*60))
    now=$(date +%s)

    echo "Monitoring job-id: $jid"
    for tid in {0..511}; do
        is_task_running "$jid" "$tid" || continue

        local out="$LOG_DIR/${JOB_NAME}.${jid}_${tid}.out"
        local err="$LOG_DIR/${JOB_NAME}.${jid}_${tid}.err"
        local m_out m_err m_last since
        m_out=$(file_mtime "$out")
        m_err=$(file_mtime "$err")
        m_last=$(( m_out > m_err ? m_out : m_err ))

        # No log yet? treat as grace period (1 interval)
        [[ $m_last -eq 0 ]] && continue

        since=$(( now - m_last ))
        if (( since > timeout_sec )); then
            echo "Task $tid stuck (idle $((since/60)) min) → requeue"
            cancel_and_requeue "$jid" "$tid"
        else
            printf "Task %-3s active (idle %2d min)\n" "$tid" $((since/60))
        fi
    done
}

# -------------------------------  Main loop  ---------------------------------
trap 'echo; echo "Monitor stopped."; exit' INT

while true; do
    [[ -z $JOB_ID ]] && JOB_ID=$(get_job_id "$JOB_NAME")
    if [[ -z $JOB_ID ]]; then
        echo "$(date): no matching job found → exiting monitor"
        break
    fi
    
    # Check if the job is still running
    if ! is_job_running "$JOB_ID"; then
        echo "$(date): job $JOB_ID is no longer running → exiting monitor"
        break
    fi
    
    check_tasks "$JOB_ID"
    echo "---- sleep $CHECK_INTERVAL s ----"
    sleep "$CHECK_INTERVAL"
done

echo "Monitor finished."
