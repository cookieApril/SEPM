#!/usr/bin/env bash
# Linux experiments that need ALFWorld (TextWorld), OfficeBench (Docker) or MARBLE's DB (Docker).
# Every job writes one result JSON and is skipped when that file exists, so stages can be
# re-run after interruptions. Each job keeps its own SEPM database / G-Memory directory.
#
#   bash scripts/run_linux_suite.sh <stage>      stages: smoke e1 ablation loo transfer
#                                                         sensitivity teamsize mab-db report
#   SEEDS="0" (default) | "0 1 2" (stage 2)   JOBS=8 parallel jobs   ACTOR=gpt-5-mini
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GM="$ROOT/external/GMemory"
PY="${PY:-python}"
OUT="${OUT:-$ROOT/benchmark-results/linux}"
SEEDS="${SEEDS:-0}"
JOBS="${JOBS:-8}"
ACTOR="${ACTOR:-gpt-5-mini}"
SOP="${SOP:-gpt-5-mini}"
METHODS="${METHODS:-no-memory generative metagpt gmemory sepm}"
TASKS="${TASKS:-alfworld officebench}"
set -a; [ -f "$ROOT/.env" ] && . <(tr -d '\r' < "$ROOT/.env"); set +a
unset SEPM_DB   # never share one pool across conditions
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/jobs"

# job <group> <task> <mas> <method> <seed> <ablation> [KEY=VALUE ...]  -> appends one command line
# Extra KEY=VALUE pairs come last in `env`, so they override defaults (e.g. a shared SEPM_DB).
job() {
  local group=$1 task=$2 mas=$3 method=$4 seed=$5 ablation=$6; shift 6
  local name="${task}__${mas}__${method}__${ablation}__s${seed}"
  local extra="" kv
  for kv in "$@"; do
    extra+=" '$kv'"
    case "${kv%%=*}" in SEPM_DB|SEPM_GMEMORY_WORKDIR) ;; *) name+="__${kv//[^A-Za-z0-9.=_-]/_}";; esac
  done
  local out="$OUT/$group/$name.json"
  [ -f "$out" ] && return 0
  printf '%s\n' "mkdir -p '$OUT/$group' && cd '$GM' && env SEPM_EVAL_BENCHMARK=$group SEPM_EVAL_TASK=$task \
SEPM_EVAL_METHOD=$method SEPM_EVAL_ACTOR_MODEL=$ACTOR SEPM_EVAL_SOP_MODEL=$SOP SEPM_EVAL_MAS=$mas \
SEPM_EVAL_SEED=$seed SEPM_EVAL_ABLATION=$ablation SEPM_EVAL_OUTPUT='$out' \
SEPM_EVAL_ACTOR_BASE_URL=\"\$OPENAI_BASE_URL\" SEPM_EVAL_ACTOR_API_KEY=\"\$OPENAI_API_KEY\" \
SEPM_EVAL_SOP_BASE_URL=\"\$OPENAI_BASE_URL\" SEPM_EVAL_SOP_API_KEY=\"\$OPENAI_API_KEY\" \
SEPM_DB='${out%.json}.sepm.db'$extra \
$PY '$ROOT/adapters/sepm_gmemory_adapter.py' --task $task --mas $mas --actor-model $ACTOR \
--sop-model $SOP --memory-method $method --seed $seed --output '$out' > '${out%.json}.log' 2>&1 \
&& echo 'DONE $name' || echo 'FAILED $name (see ${out%.json}.log)'" >> "$JOBFILE"
}

run_jobs() {
  local n; n=$(wc -l < "$JOBFILE")
  echo "[$STAGE] $n pending jobs, $JOBS in parallel"
  [ "$n" -gt 0 ] && tr '\n' '\0' < "$JOBFILE" | xargs -0 -P "$JOBS" -I{} bash -c '{}'
}

STAGE="${1:-}"
JOBFILE="$OUT/jobs/$STAGE.txt"; : > "$JOBFILE"
case "$STAGE" in
  smoke)  # one ALFWorld and one OfficeBench case per method (quick wiring check)
    for t in $TASKS; do for m in $METHODS; do
      case $t in alfworld) cid=0;; officebench) cid=1-1/0;; esac
      job smoke "$t" autogen "$m" 0 full "SEPM_EVAL_CASE_ID=$cid"; done; done ;;
  e1)     # Table 1 / Fig. 6 / Tables S13, S14, S16 (AutoGen + DyLAN)
    for s in $SEEDS; do for t in $TASKS; do for mas in autogen dylan; do for m in $METHODS; do
      job e1 "$t" "$mas" "$m" "$s" full; done; done; done; done ;;
  ablation)  # Figs. 7-8 (AutoGen). "No added components" = e1 no-memory, "Full" = e1 sepm.
    for s in $SEEDS; do for t in $TASKS; do
      for a in blackboard-only sop-only divergence-only textual-procedures; do
        job ablation "$t" autogen sepm "$s" "$a"; done; done; done ;;
  loo)    # Fig. 5a: ALFWorld leave-one-type-out; build on 5 types, test frozen on the 6th
    for ty in put clean heat cool examine puttwo; do for m in sepm gmemory; do
      pool="$OUT/loo/pools/${m}__without_${ty}"; mkdir -p "$pool"
      job loo alfworld autogen "$m" 0 full "SEPM_EXCLUDE_TASK_TYPES=$ty" "SEPM_GMEMORY_WORKDIR=$pool" \
          "SEPM_DB=$pool/sepm.db"
    done; done
    run_jobs; : > "$JOBFILE"
    for ty in put clean heat cool examine puttwo; do for m in sepm gmemory; do
      pool="$OUT/loo/pools/${m}__without_${ty}"
      job loo-test alfworld autogen "$m" 0 full "SEPM_TASK_TYPES=$ty" "SEPM_FREEZE_POOL=1" \
          "SEPM_GMEMORY_WORKDIR=$pool" "SEPM_DB=$pool/sepm.db"
    done; done
    ;;
  transfer)  # Fig. 5b: frozen AutoGen pool used under DyLAN (needs e1 seed-0 SEPM AutoGen runs)
    for t in $TASKS; do
      src="$OUT/e1/${t}__autogen__sepm__full__s0.sepm.db"
      [ -f "$src" ] || { echo "missing $src (run e1 first)"; continue; }
      cp -n "$src" "$OUT/transfer_${t}_autogen_pool.db" 2>/dev/null
      job transfer "$t" dylan sepm 0 full "SEPM_FREEZE_POOL=1" "SEPM_DB=$OUT/transfer_${t}_autogen_pool.db"
    done
    ;;
  sensitivity)  # Fig. S3: one alignment hyperparameter at a time (defaults are e1 sepm)
    for t in $TASKS; do
      for kv in goal_divergence_threshold:0.25 goal_divergence_threshold:0.45 \
                plan_divergence_threshold:0.30 plan_divergence_threshold:0.50 \
                goal_embedding_weight:0.35 goal_embedding_weight:0.75 \
                state_confidence_threshold:0.60 state_confidence_threshold:0.80; do
        job sensitivity "$t" autogen sepm 0 full "SEPM_CONFIG_OVERRIDES={\"${kv%%:*}\":${kv##*:}}"
      done; done ;;
  teamsize)  # Table S17. NOTE: only DyLAN exposes a team size (node_num) in G-Memory; see docs/BENCHMARKS.md.
    for t in $TASKS; do for n in 2 3 5; do for m in no-memory gmemory sepm; do
      job teamsize "$t" dylan "$m" 0 full "SEPM_AGENT_COUNT=$n"; done; done; done ;;
  mab-db)   # MultiAgentBench database (Docker Postgres; tasks run one at a time)
    cd "$ROOT" && $PY -m sepm.bench.marble_mab --conditions \
      database:no-memory database:generative database:metagpt database:gmemory database:sepm \
      --out "$ROOT/benchmark-results/multiagentbench" --rerun-errors; exit $? ;;
  report)
    cd "$ROOT" && $PY -m sepm.bench.report --linux "$OUT"; exit $? ;;
  *) echo "usage: $0 {smoke|e1|ablation|loo|transfer|sensitivity|teamsize|mab-db|report}"; exit 64 ;;
esac
run_jobs
