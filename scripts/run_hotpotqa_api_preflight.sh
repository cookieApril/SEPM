#!/usr/bin/env bash
# One bounded paid-API HotpotQA smoke after the local chain has passed.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${TEAM_MEMORY_ENV_FILE:-$ROOT_DIR/.env}"
RUN_ID="${HOTPOTQA_API_RUN_ID:-v2}"
case "$RUN_ID" in
  *[!a-zA-Z0-9._-]*|'') echo "Invalid HOTPOTQA_API_RUN_ID: $RUN_ID" >&2; exit 64 ;;
esac
OUTPUT="$ROOT_DIR/benchmark-results/hotpotqa-api-preflight/results/gpt-5-mini-single-$RUN_ID.json"
LOCK="$ROOT_DIR/benchmark-results/hotpotqa-api-preflight/single-$RUN_ID.lock"
CASE_ID="5a8b57f25542995d1e6f1371"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing API environment file: $ENV_FILE" >&2
  exit 78
fi
load_first_env_value() {
  local wanted="$1" raw_key raw_value
  while IFS='=' read -r raw_key raw_value; do
    raw_key="${raw_key//[[:space:]]/}"
    [[ "$raw_key" == "$wanted" ]] || continue
    raw_value="${raw_value#\"}"; raw_value="${raw_value%\"}"
    raw_value="${raw_value#\'}"; raw_value="${raw_value%\'}"
    if [[ -n "$raw_value" ]]; then
      printf '%s' "$raw_value"
      return 0
    fi
  done <"$ENV_FILE"
  return 1
}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  OPENAI_API_KEY="$(load_first_env_value OPENAI_API_KEY || true)"
fi
if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  OPENAI_BASE_URL="$(load_first_env_value OPENAI_BASE_URL || true)"
fi

BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not configured" >&2
  exit 78
fi

mkdir -p "$(dirname -- "$OUTPUT")"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another HotpotQA API preflight is active" >&2
  exit 75
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TEAM_MEMORY_EVAL_BENCHMARK="hotpotqa-api-preflight"
export TEAM_MEMORY_EVAL_TASK="hotpotqa"
export TEAM_MEMORY_EVAL_METHOD="team-memory"
export TEAM_MEMORY_EVAL_ACTOR_MODEL="gpt-5-mini"
export TEAM_MEMORY_EVAL_SOP_MODEL="gpt-5-mini"
export TEAM_MEMORY_EVAL_MAS="autogen"
export TEAM_MEMORY_EVAL_SEED="0"
export TEAM_MEMORY_EVAL_ABLATION="full"
export TEAM_MEMORY_EVAL_CASE_ID="$CASE_ID"
export TEAM_MEMORY_EVAL_OUTPUT="$OUTPUT"
export TEAM_MEMORY_EVAL_ACTOR_BASE_URL="$BASE_URL"
export TEAM_MEMORY_EVAL_ACTOR_API_KEY="$OPENAI_API_KEY"
export TEAM_MEMORY_EVAL_SOP_BASE_URL="$BASE_URL"
export TEAM_MEMORY_EVAL_SOP_API_KEY="$OPENAI_API_KEY"
export TEAM_MEMORY_EVAL_ACTOR_MAX_TOKENS="1024"
export TEAM_MEMORY_EVAL_MAX_RETRIES="0"
export TEAM_MEMORY_EVAL_AGENT_ATTEMPTS="1"
export TEAM_MEMORY_DB="$ROOT_DIR/benchmark-results/hotpotqa-api-preflight/team-memory-$RUN_ID.db"

cd "$ROOT_DIR/external/GMemory"
timeout 600 conda run --no-capture-output -n GMemory \
  python "$ROOT_DIR/adapters/team_memory_gmemory_adapter.py" \
    --task hotpotqa \
    --mas autogen \
    --actor-model gpt-5-mini \
    --sop-model gpt-5-mini \
    --memory-method team-memory \
    --seed 0 \
    --output "$OUTPUT"
