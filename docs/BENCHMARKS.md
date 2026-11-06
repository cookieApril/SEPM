# SEPM Linux runbook (for the remote executor)

This runbook covers the experiments that cannot run on the Windows workstation: ALFWorld
(TextWorld is Linux-only), OfficeBench (Docker), and the MultiAgentBench **database** scenario
(Docker Postgres). ImplicitMemBench and MultiAgentBench coding/research run on Windows.

**Out of scope (do not run):** WebArena; Table 3 model sensitivity; TDW-MAT / C-WAH (need GPU);
Qwen2.5-7B / qwen3.5-9b runs; baselines without official code (LEGOMem, MATM, AWM, Memp,
Synapse, LLMA-Mem); human-annotated detection precision.

**Rules.** Never edit result files or copy numbers from `paper/`. If a stage fails, fix the
cause, delete only that job's `.json` (and `.log`) and re-run the stage; finished jobs are skipped.
Report every patch you make to third-party code.

---

## 0. Server prerequisites

* Linux x86-64, ≥ 8 cores, ≥ 32 GB RAM, ≥ 100 GB disk. No GPU needed.
* Docker usable **without sudo** for this user (`docker ps` works). MARBLE's DB environment calls
  `sudo docker compose …`; either grant passwordless sudo for docker or apply step 2.4.
* Outbound HTTPS to GitHub, PyPI, Hugging Face (or `HF_ENDPOINT=https://hf-mirror.com`),
  and the API relay in `.env` (`OPENAI_BASE_URL`, `OPENAI_API_KEY`).
* Java is **not** needed unless you also run SciWorld (we do not).

## 1. Code and environments

```bash
git clone <this repository> sepm && cd sepm          # or copy the workspace, including .env
conda create -y -n sepm python=3.12 && conda activate sepm
pip install -e ".[dev]" openai pyyaml networkx numpy tqdm tiktoken attrs scikit-learn finch-clust

mkdir -p external && cd external
git clone https://github.com/bingreeky/GMemory.git
git clone https://github.com/zlwang-cs/OfficeBench.git
git clone --depth 1 https://github.com/ulab-uiuc/MARBLE.git
cd ..
python scripts/patch_gmemory_py311.py external/GMemory   # harmless on 3.12; keeps checkouts identical
python scripts/patch_marble.py external/MARBLE           # REQUIRED: upstream evaluator.py has a syntax error

pip install -r external/GMemory/requirements.txt         # alfworld 0.3.5, textworld, langchain-chroma, sentence-transformers
pip install -r external/OfficeBench/requirements.txt docker google-generativeai pytz fire
pip install litellm beartype "ruamel.yaml" arxiv semanticscholar beautifulsoup4 PyPDF2 colorlog \
            levenshtein names requests psycopg2-binary colorama keybert
```

### 1.1 ALFWorld game files

```bash
export ALFWORLD_DATA=$HOME/.cache/alfworld
alfworld-download                                        # ~2 GB; json_2.1.1 + logic
ln -s $ALFWORLD_DATA/json_2.1.1 external/GMemory/data/alfworld/json_2.1.1
ln -s $ALFWORLD_DATA/logic      external/GMemory/data/alfworld/logic
python - <<'EOF'
import json; t=json.load(open("external/GMemory/data/alfworld/alfworld_tasks_suffix.json"))
import os; missing=[x["gamefile"] for x in t if not os.path.exists("external/GMemory/"+x["gamefile"])]
print(len(t), "tasks;", len(missing), "missing gamefiles")   # expect: 134 tasks; 0 missing
EOF
```

### 1.2 OfficeBench image

```bash
cd external/OfficeBench && docker build -t officebench -f docker/Dockerfile . && cd ../..
python - <<'EOF'
import glob; print(len(glob.glob("external/OfficeBench/tasks/*/subtasks/*.json")))   # expect 300
EOF
```

### 1.3 Sanity checks

```bash
pytest -q tests/test_plugins.py tests/test_gmemory_adapter.py tests/test_core.py   # all pass
python -c "import alfworld, textworld; print('alfworld ok')"
docker run --rm officebench echo ok
```

## 2. Known upstream issues and fixes

1. **MARBLE `evaluator.py` syntax error** → `scripts/patch_marble.py` (step 1).
2. **MARBLE Graph communication/planning prompts** contain literal `{"rating": X}` and break
   `str.format`; the harness fills them by placeholder substitution (`marble_mab._graph_rating`).
3. **G-Memory imports SciWorld/PDDL at load**; their task files ship in `external/GMemory/data`.
   If `import scienceworld` fails, `pip install scienceworld==1.2.2` (Java only needed to *run* it).
4. **MARBLE DB environment uses `sudo docker …` and `docker logs -tf`** (`-f` follows the log and
   can block). If either blocks, patch `marble/environments/db_env.py`: drop `sudo` and replace
   `-tf` with `-t --tail 500`; record the diff in your report.
5. **tiktoken downloads** (`o200k_base`, `cl100k_base`) on first use; pre-fetch once:
   `python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('o200k_base','cl100k_base')]"`.

## 3. Stages

All stages: `bash scripts/run_linux_suite.sh <stage>`; results in `benchmark-results/linux/<group>/`,
one JSON per job (+ `.log`, + `.sepm.db` for SEPM). Tunables: `JOBS` (parallel jobs, default 8),
`SEEDS` (default `0`), `TASKS` (default `alfworld officebench`), `METHODS`.

| Stage | Paper item | Jobs (seed 0) | Notes |
|---|---|---|---|
| `smoke` | wiring check | 10 | one case per method/benchmark; **run first, inspect logs** |
| `e1` | Table 1, Fig. 4, Fig. 6, Tables 9/10/13/14 | 20 | AutoGen+DyLAN × 5 memories × 2 benchmarks |
| `ablation` | Fig. 7, Table 12 | 8 | blackboard-only, sop-only (=BB+memory), divergence-only (=BB+alignment), textual-procedures |
| `loo` | Fig. 9a | 12+12 | builds pools on 5 ALFWorld types, then tests frozen on the 6th |
| `transfer` | Fig. 9b | 2 | needs `e1` AutoGen SEPM seed-0 pools |
| `sensitivity` | Fig. 10 | 16 | τ_g, τ_p, β, evidence confidence (one at a time) |
| `teamsize` | Table 11 | 18 | **see 3.1** |
| `mab-db` | Table 6 (database) | 5×100 tasks | one DB task at a time (shared Postgres) |
| `report` | all | – | writes `benchmark-results/REPORT.md` |

Recommended order: `smoke` → `e1` → `ablation` + `sensitivity` (can overlap) → `loo` →
`transfer` → `teamsize` → `mab-db` → `report`. After review, stage 2 adds seeds:
`SEEDS="1 2" bash scripts/run_linux_suite.sh e1` and `... ablation` (seeds 1, 2 shuffle the order).

Rough cost (gpt-5-mini via the relay): ALFWorld ≈ 4–6 M tokens per 134-task run, OfficeBench
≈ 8–11 M per 300-task run; seed-0 `e1` ≈ 150–200 M tokens. ALFWorld runs take ~3–6 h each,
OfficeBench ~6–10 h each; `JOBS` mostly limits RAM (each ALFWorld process ≈ 1–2 GB).

### 3.1 Team size (Table 11) — needs a decision

G-Memory's AutoGen workflow has fixed roles (solver + ground truth), so it has no team-size knob.
The `teamsize` stage therefore varies DyLAN's `node_num` (2/3/5). Report it as DyLAN, not AutoGen,
unless the user asks for a modified AutoGen host.

### 3.2 What to check in `smoke`

For each job log: the run reaches `DONE`; the JSON has `metrics.case_count == 1`; for `sepm`,
`metrics.sop_model_calls > 0` after a success and `metadata.pool_statistics` is present; for
`gmemory/generative/metagpt`, the log shows memory retrieval without errors; for OfficeBench, a
`sepm_outputs/<task>/<subtask>/testbed` folder exists and evaluation printed pass/fail.

## 4. What to send back

* `benchmark-results/linux/` (JSONs and logs; `.sepm.db` files optional, they can be large),
* `benchmark-results/multiagentbench/database/`,
* `benchmark-results/REPORT.md`,
* a short note listing every third-party patch, every failed/rerun job and why, and the
  server spec and wall time per stage.
