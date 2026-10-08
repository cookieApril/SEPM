#!/usr/bin/env bash
# Run paper stages on Linux; reruns skip successful cells through SQLite checkpoints.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PHASE="${1:-all}"
CONFIG="${CONFIG:-evaluation_matrix.json}"
ACTOR_MODEL="${ACTOR_MODEL:-gpt-5-mini}"
SOP_MODEL="${SOP_MODEL:-gpt-5-mini}"
SOP_MODEL_FILTER="${SOP_MODEL_FILTER:-}"
SMOKE_MODEL_FILTER="${SMOKE_MODEL_FILTER:-}"
ABLATION_MAS="${ABLATION_MAS:-autogen}"
RERUN_FAILED="${RERUN_FAILED:-0}"
PAPER_RUN_LOCK_PATH="${PAPER_RUN_LOCK_PATH:-benchmark-results/unified-v3/paper-run.lock}"

mkdir -p benchmark-results/unified-v3 benchmark-results/logs
EVAL_CMD=(python -m sepm.evaluation_runner)
ABLATION_CMD=(python -m sepm.ablation.runner)
TABLES_CMD=(python -m sepm.paper_tables)
export PYTHONPATH="${PYTHONPATH:-src}"

# Prevent two full runs from writing the same SQLite checkpoint. Without flock,
# the user must ensure that only one run_paper_experiments.sh uses this workspace.
if command -v flock >/dev/null 2>&1; then
  mkdir -p "$(dirname -- "$PAPER_RUN_LOCK_PATH")"
  exec 9>"$PAPER_RUN_LOCK_PATH"
  if ! flock -n 9; then
    echo "another paper experiment process is already using this workspace" >&2
    exit 3
  fi
fi

retry_args=()
if [[ "$RERUN_FAILED" == "1" ]]; then
  retry_args+=(--rerun-failed)
fi

run_smoke() {
  if [[ -n "$SMOKE_MODEL_FILTER" ]]; then
    "${EVAL_CMD[@]}" smoke --config "$CONFIG" --model "$SMOKE_MODEL_FILTER"
    return
  fi
  # Read enabled models from the matrix so new SOP models receive connectivity checks.
  while IFS= read -r model_id; do
    "${EVAL_CMD[@]}" smoke --config "$CONFIG" --model "$model_id"
  done < <(
    python -c 'import json,sys; c=json.load(open(sys.argv[1], encoding="utf-8")); print("\n".join(m["id"] for m in c["models"] if m.get("enabled", True)))' "$CONFIG"
  )
}

require_external_jobs() {
  local benchmark_args=()
  local benchmark
  for benchmark in "$@"; do
    benchmark_args+=(--benchmark "$benchmark")
  done
  python scripts/validate_paper_setup.py \
    --config "$CONFIG" --require-jobs "${benchmark_args[@]}"
}

write_matrices() {
  mkdir -p benchmark-results/plans
  "${EVAL_CMD[@]}" matrix --config "$CONFIG" \
    --benchmark cross-benchmark-generality \
    --actor-model "$ACTOR_MODEL" --sop-model "$SOP_MODEL" \
    > benchmark-results/plans/cross-benchmark-generality.jsonl
  "${EVAL_CMD[@]}" matrix --config "$CONFIG" \
    --benchmark sop-model-sensitivity --actor-model "$ACTOR_MODEL" \
    > benchmark-results/plans/sop-model-sensitivity.jsonl
  "${ABLATION_CMD[@]}" matrix --config "$CONFIG" \
    --benchmark component-ablation --mas "$ABLATION_MAS" \
    --actor-model "$ACTOR_MODEL" --sop-model "$SOP_MODEL" \
    > benchmark-results/plans/ablation.jsonl
}

run_generality() {
  "${EVAL_CMD[@]}" run --config "$CONFIG" \
    --benchmark cross-benchmark-generality \
    --memory-method sepm \
    --actor-model "$ACTOR_MODEL" --sop-model "$SOP_MODEL" \
    "${retry_args[@]}"
}

run_sop_models() {
  # ScienceWorld launches a JVM from the GMemory environment. Honor an
  # explicitly configured JDK and make it visible to conda-run children.
  if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/java" ]]; then
    export PATH="$JAVA_HOME/bin:$PATH"
  fi
  local model_args=()
  if [[ -n "$SOP_MODEL_FILTER" ]]; then
    model_args+=(--sop-model "$SOP_MODEL_FILTER")
  fi
  "${EVAL_CMD[@]}" run --config "$CONFIG" \
    --benchmark sop-model-sensitivity --actor-model "$ACTOR_MODEL" \
    "${model_args[@]}" \
    "${retry_args[@]}"
}

run_ablation() {
  "${ABLATION_CMD[@]}" run --config "$CONFIG" \
    --benchmark component-ablation --mas "$ABLATION_MAS" \
    --actor-model "$ACTOR_MODEL" --sop-model "$SOP_MODEL" \
    "${retry_args[@]}"
}

build_outputs() {
  "${EVAL_CMD[@]}" report --config "$CONFIG"
  "${TABLES_CMD[@]}" \
    --summary benchmark-results/unified-v3/summary.json \
    --actor-model "$ACTOR_MODEL" --sop-model "$SOP_MODEL" \
    --mas "$ABLATION_MAS"
}

case "$PHASE" in
  preflight)
    python scripts/validate_paper_setup.py --config "$CONFIG"
    ;;
  smoke) run_smoke ;;
  matrices) write_matrices ;;
  generality) require_external_jobs cross-benchmark-generality; run_generality ;;
  sop-models) require_external_jobs sop-model-sensitivity; run_sop_models ;;
  ablation) require_external_jobs component-ablation; run_ablation ;;
  report) build_outputs ;;
  all)
    require_external_jobs cross-benchmark-generality sop-model-sensitivity component-ablation
    run_smoke
    write_matrices
    run_generality
    run_sop_models
    run_ablation
    build_outputs
    ;;
  *)
    echo "usage: $0 {preflight|smoke|matrices|generality|sop-models|ablation|report|all}" >&2
    exit 64
    ;;
esac
