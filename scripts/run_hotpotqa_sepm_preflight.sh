#!/usr/bin/env bash
# One local-only HotpotQA case through the genuine GMemory + SEPM adapter.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${HOTPOTQA_MODEL:-Qwen/Qwen3.5-2B}"
BASE_URL="${HOTPOTQA_BASE_URL:-http://127.0.0.1:8113/v1}"
CASE_ID="${HOTPOTQA_CASE_ID:-5a8b57f25542995d1e6f1371}"
RUN_ID="${HOTPOTQA_RUN_ID:-v2}"
case "$RUN_ID" in
  *[!a-zA-Z0-9._-]*|'') echo "Invalid HOTPOTQA_RUN_ID: $RUN_ID" >&2; exit 64 ;;
esac
OUTPUT="$ROOT_DIR/benchmark-results/hotpotqa-local/results/sepm-single-$RUN_ID.json"

case "$BASE_URL" in
  http://127.0.0.1:*|http://localhost:*) ;;
  *) echo "Refusing non-local HotpotQA preflight endpoint: $BASE_URL" >&2; exit 64 ;;
esac

curl -fsS --max-time 5 "${BASE_URL%/v1}/v1/models" >/dev/null
mkdir -p "$(dirname -- "$OUTPUT")"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export SEPM_EVAL_BENCHMARK="hotpotqa-local-preflight"
export SEPM_EVAL_TASK="hotpotqa"
export SEPM_EVAL_METHOD="sepm"
export SEPM_EVAL_ACTOR_MODEL="$MODEL"
export SEPM_EVAL_SOP_MODEL="$MODEL"
export SEPM_EVAL_MAS="autogen"
export SEPM_EVAL_SEED="0"
export SEPM_EVAL_ABLATION="full"
export SEPM_EVAL_CASE_ID="$CASE_ID"
export SEPM_EVAL_OUTPUT="$OUTPUT"
export SEPM_EVAL_ACTOR_BASE_URL="$BASE_URL"
export SEPM_EVAL_ACTOR_API_KEY="EMPTY"
export SEPM_EVAL_SOP_BASE_URL="$BASE_URL"
export SEPM_EVAL_SOP_API_KEY="EMPTY"
export SEPM_EVAL_ACTOR_MAX_TOKENS="256"
export SEPM_DB="$ROOT_DIR/benchmark-results/hotpotqa-local/sepm-$RUN_ID.db"

cd "$ROOT_DIR/external/GMemory"
conda run --no-capture-output -n GMemory \
  python "$ROOT_DIR/adapters/sepm_gmemory_adapter.py" \
    --task hotpotqa \
    --mas autogen \
    --actor-model "$MODEL" \
    --sop-model "$MODEL" \
    --memory-method sepm \
    --seed 0 \
    --output "$OUTPUT"
