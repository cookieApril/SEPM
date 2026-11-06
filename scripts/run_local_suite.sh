#!/usr/bin/env bash
# Local (Windows/Git Bash or Linux) LLM-only experiments: ImplicitMemBench and MultiAgentBench coding/research.
# Every command resumes: finished items are skipped, errored items are retried (3 passes).
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=python
export PYTHONIOENCODING=utf-8 MEM0_TELEMETRY=False
WORKERS="${WORKERS:-16}"
imb() { for pass in 1 2 3; do "$PY" -m sepm.bench.implicitmem --workers "$WORKERS" "$@"; done; }

case "${1:-main}" in
  main)   # Table 3a (gpt-5-mini rows)
    for m in single-llm no-memory sepm gmemory generative metagpt mem0; do imb --method "$m"; done ;;
  interference)  # Fig. S2: procedural FTA vs interference length
    for n in 200 1000 2000; do for m in single-llm no-memory gmemory sepm; do
      imb --method "$m" --categories procedural_memory --interference "$n"; done; done ;;
  accumulate)    # Table S12: pool accumulated across items (sequential by design)
    imb --method sepm --pool accumulate --order original
    imb --method sepm --pool accumulate --order shuffled ;;
  mab)  # Table S11: MultiAgentBench coding + research; each condition is one sequential chain
    conds=""; for s in coding research; do for m in no-memory generative metagpt gmemory sepm; do conds+=" $s:$m"; done; done
    for pass in 1 2; do "$PY" -m sepm.bench.marble_mab --conditions $conds --rerun-errors; done ;;
  *) echo "usage: $0 {main|interference|accumulate|mab}"; exit 64 ;;
esac
