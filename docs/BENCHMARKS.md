# Running the Benchmarks

This guide covers the benchmark harnesses shipped with SEPM. ALFWorld (TextWorld), OfficeBench
(Docker), and the MultiAgentBench **database** scenario (Docker Postgres) require Linux.
ImplicitMemBench and MultiAgentBench coding/research also run on Windows (Git Bash).

Every job writes one result file and is skipped when that file already exists, so any stage can be
re-run after an interruption. To redo a job, delete its `.json` (and `.log`) and re-run the stage.

---

## 0. Prerequisites

* Linux x86-64, 8+ cores, 32 GB RAM, 100 GB disk. No GPU is needed for the stages below.
* Docker usable without `sudo` (`docker ps` works). MARBLE's database environment calls
  `sudo docker compose …`; either allow passwordless `sudo` for Docker or apply fix 2.4.
* Outbound HTTPS to GitHub, PyPI, Hugging Face, and an OpenAI-compatible endpoint configured in
  `.env` (`OPENAI_BASE_URL`, `OPENAI_API_KEY`; see [`.env.example`](../.env.example)).

## 1. Code and environments

```bash
conda create -y -n sepm python=3.12 && conda activate sepm
pip install -e ".[dev,eval]" networkx tqdm attrs scikit-learn finch-clust

mkdir -p external && cd external
git clone https://github.com/bingreeky/GMemory.git
git clone https://github.com/zlwang-cs/OfficeBench.git
git clone --depth 1 https://github.com/ulab-uiuc/MARBLE.git
git clone https://github.com/chonghanqin/ImplicitMemBench.git
cd ..
python scripts/patch_gmemory_py311.py external/GMemory   # PEP 701 f-strings -> Python 3.11 syntax
python scripts/patch_marble.py external/MARBLE           # required: upstream evaluator.py has a syntax error

pip install -r external/GMemory/requirements.txt
pip install -r external/OfficeBench/requirements.txt docker google-generativeai pytz fire
pip install litellm beartype "ruamel.yaml" arxiv semanticscholar beautifulsoup4 PyPDF2 colorlog \
            levenshtein names requests psycopg2-binary colorama keybert mem0ai
```

### 1.1 ALFWorld game files

```bash
export ALFWORLD_DATA=$HOME/.cache/alfworld
alfworld-download
ln -s $ALFWORLD_DATA/json_2.1.1 external/GMemory/data/alfworld/json_2.1.1
ln -s $ALFWORLD_DATA/logic      external/GMemory/data/alfworld/logic
python - <<'EOF'
import json, os
t = json.load(open("external/GMemory/data/alfworld/alfworld_tasks_suffix.json"))
missing = [x["gamefile"] for x in t if not os.path.exists("external/GMemory/" + x["gamefile"])]
print(len(t), "tasks;", len(missing), "missing gamefiles")   # expected: 134 tasks; 0 missing
EOF
```

### 1.2 OfficeBench image

```bash
cd external/OfficeBench && docker build -t officebench -f docker/Dockerfile . && cd ../..
python -c "import glob; print(len(glob.glob('external/OfficeBench/tasks/*/subtasks/*.json')))"   # 300
```

### 1.3 Sanity checks

```bash
pytest -q
python scripts/validate_paper_setup.py
python -c "import alfworld, textworld; print('alfworld ok')"
docker run --rm officebench echo ok
```

## 2. Known upstream issues

1. **MARBLE `evaluator.py` syntax error.** Fixed by `scripts/patch_marble.py`.
2. **MARBLE communication/planning prompts** contain literal `{"rating": X}` and break
   `str.format`; the harness fills the named placeholders instead (`marble_mab._graph_rating`).
3. **G-Memory imports SciWorld/PDDL at load time.** If `import scienceworld` fails, run
   `pip install scienceworld==1.2.2` (Java is only needed to run SciWorld itself).
4. **MARBLE's database environment uses `sudo docker …` and `docker logs -tf`**, which can block.
   If so, edit `marble/environments/db_env.py`: drop `sudo` and replace `-tf` with `-t --tail 500`.
5. **tiktoken encodings** are downloaded on first use. Pre-fetch them once:
   `python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('o200k_base','cl100k_base')]"`.

## 3. Linux stages

Run `bash scripts/run_linux_suite.sh <stage>`. Results go to `benchmark-results/linux/<group>/`,
one JSON per job plus its `.log` and, for SEPM, its `.sepm.db` procedure pool.
Environment variables: `JOBS` (parallel jobs, default 8), `SEEDS` (default `0`; `"0 1 2"` for the
three shuffled task orders), `TASKS` (default `alfworld officebench`), `METHODS`, `ACTOR`, `SOP`.

| Stage | Paper item | Jobs (seed 0) | Notes |
|---|---|---|---|
| `smoke` | wiring check | 10 | one case per method and benchmark; run first and inspect the logs |
| `e1` | Table 1, Fig. 6, Tables S13, S14, S16 | 20 | AutoGen and DyLAN × 5 memories × 2 benchmarks |
| `ablation` | Figs. 7–8 | 8 | blackboard only, blackboard + memory, blackboard + alignment, textual procedures |
| `loo` | Fig. 5a | 12 + 12 | builds pools on five ALFWorld task types, then tests them frozen on the sixth |
| `transfer` | Fig. 5b | 2 | needs the seed-0 AutoGen SEPM pools from `e1` |
| `sensitivity` | Fig. S3 | 16 | δ<sub>g</sub>, δ<sub>p</sub>, β, and evidence confidence, one at a time |
| `teamsize` | Table S17 | 18 | varies the number of task agents (see 3.1) |
| `mab-db` | Table S11 (database) | 5 × 100 tasks | one task at a time on a shared Postgres |
| `report` | all | – | writes `benchmark-results/REPORT.md` |

Recommended order: `smoke` → `e1` → `ablation` and `sensitivity` → `loo` → `transfer` →
`teamsize` → `mab-db` → `report`.

Approximate cost with `gpt-5-mini`: 4–6 M tokens per 134-task ALFWorld run and 8–11 M tokens per
300-task OfficeBench run. Each ALFWorld process needs about 1–2 GB of RAM, so `JOBS` is mostly
limited by memory.

### 3.1 Team size

G-Memory's AutoGen workflow has fixed roles, so the `teamsize` stage varies the agent count
through DyLAN's `node_num` (2, 3, 5).

### 3.2 Checking the smoke stage

For each job log: the run reaches `DONE`, and the JSON has `metrics.case_count == 1`. For `sepm`,
`metrics.sop_model_calls > 0` after a success and `metadata.pool_statistics` is present. For the
G-Memory family, the log shows memory retrieval without errors. For OfficeBench, a
`sepm_outputs/<task>/<subtask>/testbed` folder exists and the evaluator printed pass or fail.

## 4. LLM-only benchmarks (any OS)

```bash
bash scripts/run_local_suite.sh main           # ImplicitMemBench, gpt-5-mini rows of Table 3a
bash scripts/run_local_suite.sh interference   # Fig. S2
bash scripts/run_local_suite.sh accumulate     # Table S12
bash scripts/run_local_suite.sh mab            # MultiAgentBench coding + research (Table S11)
python -m sepm.bench.report                    # collects everything into benchmark-results/REPORT.md
```
