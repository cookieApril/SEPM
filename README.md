<h1 align="center">SEPM: Self-Evolving Procedural Memory</h1>

<p align="center"><strong>Task-graph-grounded memory for multi-agent collaboration</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-ICLR%202027%20submission-b5212f" alt="ICLR 2027 submission">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-6b7280" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> ·
  <a href="#-main-results">Results</a> ·
  <a href="#-project-structure">Code</a> ·
  <a href="#-quick-start">Quick start</a> ·
  <a href="#-evaluation">Evaluation</a>
</p>

---

## 👀 Overview

Multi-agent teams need more than a list of successful actions. A useful procedure also records **which agent depends on another, what evidence completes a step, and when a handoff is safe**. SEPM uses a task dependency graph as the shared structure for both live coordination and future procedural reuse.

The implementation connects three memory layers: private agent context, a shared blackboard of task and execution evidence, and versioned procedural memories (SOPs). During a task, SEPM checks goal, plan, and state divergence; after a task, it can turn verified actions and dependencies into a reusable procedure. Candidates pass evidence, utility, and safety gates before publication.

<p align="center">
  <img src="docs/readme-assets/framework.png" width="94%" alt="SEPM architecture with task graph, blackboard, divergence detection, and evolving procedures"><br>
  <em>Framework overview (paper Figure 2). Task graphs connect planning, shared execution records, alignment, and procedure revision.</em>
</p>

### What SEPM implements

| Mechanism | Role in the workflow | Code |
| --- | --- | --- |
| Shared task graph and blackboard | Store assignments, prerequisites, observations, outcomes, errors, and evidence | [`models.py`](src/sepm/models.py), [`storage.py`](src/sepm/storage.py) |
| Divergence and realignment | Detect goal, plan, and state disagreements; resolve state claims by evidence priority | [`divergence.py`](src/sepm/divergence.py), [`evidence.py`](src/sepm/evidence.py) |
| Evolving procedures | Propose, validate, reproduce, and version graph-structured SOPs | [`service.py`](src/sepm/service.py), [`safety.py`](src/sepm/safety.py) |
| Contextual retrieval | Filter applicable SOPs and rank them with lexical and vector relevance | [`retrieval.py`](src/sepm/retrieval.py) |

The [storage schema](docs/STORAGE_SCHEMA.md) describes the graph objects, blackboard events, and SQLite version history.

<p align="center">
  <img src="docs/readme-assets/case-study.png" width="92%" alt="Cleaning and storage case illustrating plan and state divergence"><br>
  <em>Illustrative case (paper Figure 3). The team redirects an unrelated action and verifies cleaning before handing items to the storage agent; the revised procedure keeps that prerequisite.</em>
</p>

---

## 📊 Main Results

The figures and numbers in this section are **reported in the supplied SEPM manuscript**. Its main evaluation uses 134 ALFWorld, 812 WebArena, and 300 OfficeBench tasks per condition, with one base-seed run. [Evaluation](#-evaluation) explains the scope of the code and local experiment matrix in this workspace.

### Cross-benchmark task performance

The paper reports the highest task score for SEPM in all six benchmark–host settings. Under AutoGen, the gains over the same host without added memory are **+29.85 points** on ALFWorld, **+25.86** on WebArena, and **+13.00** on OfficeBench.

| Host | Memory | ALFWorld ↑ | WebArena ↑ | OfficeBench ↑ |
| --- | --- | ---: | ---: | ---: |
| AutoGen | No added memory | 59.70 | 12.44 | 41.67 |
| AutoGen | Generative Agents | 85.82 | 15.27 | 38.33 |
| AutoGen | MetaGPT | 78.36 | 25.99 | 45.33 |
| AutoGen | G-Memory | 83.58 | 30.30 | 50.00 |
| AutoGen | **SEPM** | **89.55** | **38.30** | **54.67** |
| DyLAN | No added memory | 53.73 | 14.90 | 33.33 |
| DyLAN | Generative Agents | 65.67 | 15.27 | 33.00 |
| DyLAN | MetaGPT | 49.25 | 22.41 | 35.00 |
| DyLAN | G-Memory | 61.94 | 25.49 | 42.67 |
| DyLAN | **SEPM** | **68.66** | **29.80** | **47.33** |

<sub>Paper Table 1. Task scores are percentages; bold marks the best observed score within each host and benchmark.</sub>

<p align="center">
  <img src="docs/readme-assets/overall-performance.png" width="82%" alt="Average task scores for memory methods under AutoGen and DyLAN"><br>
  <em>Paper Figure 4(a). The unweighted mean of the three displayed benchmarks rises from 37.94% to 60.84% under AutoGen and from 33.99% to 48.60% under DyLAN.</em>
</p>

### Why explicit dependency graphs matter

The manuscript compares full SEPM with a textual procedure control that retains the step descriptions and prerequisites but replaces graph objects and adjacency relations with text. The reported graph-based gains are **17.16**, **8.13**, and **12.34** points on ALFWorld, WebArena, and OfficeBench.

<p align="center">
  <img src="docs/readme-assets/graph-vs-text.png" width="88%" alt="Task score comparison between textual SOPs and explicit dependency graphs"><br>
  <em>Paper Figure 5(a). Explicit graphs make prerequisite checks, plan comparison, and correction propagation operate over the same dependency structure.</em>
</p>

### Component contributions

The shared blackboard is the foundation for both memory and alignment in this ablation. The full configuration leads on all three tasks; the individual gains differ by environment.

| AutoGen configuration | ALFWorld ↑ | WebArena ↑ | OfficeBench ↑ |
| --- | ---: | ---: | ---: |
| No added components | 59.70 | 12.44 | 41.67 |
| Blackboard only | 69.40 | 27.71 | 45.00 |
| Blackboard + memory | 86.57 | 33.99 | 45.33 |
| Blackboard + alignment | 82.09 | 34.48 | 51.00 |
| **Full SEPM** | **89.55** | **38.30** | **54.67** |

<sub>Paper Figure 5(b). Task scores are percentages. Memory includes cross-task procedure extraction, revision, and reuse; alignment detects and resolves divergence.</sub>

### MultiAgentBench: task quality and communication

SEPM also reports the highest Task Score (TS) and Communication Score (CS) across the coding, research, and database environments under AutoGen.

<p align="center">
  <img src="docs/readme-assets/multiagentbench.png" width="86%" alt="MultiAgentBench task and communication scores across coding, research, and database"><br>
  <em>Paper Figure 1. TS measures task output; CS measures communication and planning. Both retain the benchmark's reporting scale.</em>
</p>

| Method | Coding TS / CS | Research TS / CS | Database TS / CS |
| --- | ---: | ---: | ---: |
| No added memory | 54.90 / 39.81 | 62.40 / 55.16 | 64.93 / 62.43 |
| Generative Agents | 55.74 / 46.09 | 61.16 / 64.64 | 65.21 / 77.16 |
| MetaGPT | 57.62 / 47.23 | 70.11 / 62.27 | 70.14 / 80.29 |
| G-Memory | 64.20 / 51.17 | 74.29 / 77.77 | 69.73 / 88.20 |
| **SEPM** | **69.09 / 60.64** | **78.37 / 88.91** | **77.03 / 91.94** |

<sub>Paper Table 3. The manuscript evaluates 100 tasks in each environment; these scores are separate from the three-benchmark mean above.</sub>

### Performance and token cost

On the 134 ALFWorld tasks under AutoGen, the paper reports **89.55%** success using **5.2 million** input and output tokens. This is higher success with fewer tokens than the reported Generative Agents (85.82%, 5.6M) and G-Memory (83.58%, 5.7M) runs. No added memory uses 4.4M tokens at 59.70% success.

<p align="center">
  <img src="docs/readme-assets/token-cost.png" width="80%" alt="ALFWorld success versus total token consumption under AutoGen"><br>
  <em>Paper Figure 4(b). Success and token totals refer to the same complete ALFWorld runs.</em>
</p>

<details>
<summary><strong>Model sensitivity — all paper Table 2 scores</strong></summary>

The paper jointly changes the task-execution and procedural-memory endpoint while keeping AutoGen, the planning agent, and the goal-consistency judge fixed. These rows therefore measure the **joint configuration**.

| Model endpoint | ALFWorld ↑ | WebArena ↑ | OfficeBench ↑ |
| --- | ---: | ---: | ---: |
| qwen3.5-0.8b | 58.21 | 14.53 | 45.67 |
| qwen3.5-2b | 57.46 | 14.78 | 45.67 |
| qwen3.5-9b | 67.16 | 16.01 | 49.67 |
| qwen3.5-27b | 81.34 | 30.42 | 55.67 |
| gemma-4-12b-it | 72.39 | 22.54 | 53.00 |
| gemma-4-31b-it | 77.61 | 28.69 | 54.33 |
| deepseek-v4-flash-0731 | **93.28** | 38.05 | 64.33 |
| qwen3.8-max | 89.55 | 39.41 | 63.67 |
| glm-5.2 | 85.82 | 37.93 | 60.67 |
| kimi-k3 | 90.30 | 42.36 | 65.33 |
| gpt-5-mini (main) | 89.55 | 38.30 | 54.67 |
| gpt-5.6-terra | 91.79 | **43.23** | **67.67** |
| claude-opus-5 | 92.54 | 42.49 | 65.33 |

<sub>Paper Table 2. Task scores are percentages; bold marks the highest observed value in each column.</sub>

</details>

---

## 📁 Project Structure

```text
SEPM/
├── src/sepm/             # Core memory service, graphs, alignment, retrieval, storage
├── adapters/             # GMemory and cross-benchmark integration bridges
├── configs/              # Adapter and benchmark configuration
├── manifests/            # Selected case IDs for local experiment plans
├── scripts/              # Setup checks and staged experiment runner
├── examples/             # Small end-to-end Python example
├── tests/                # Core and integration-contract tests
├── docs/                 # Storage schema and README figure assets
├── evaluation_matrix.json
├── environment.yml       # Optional full Conda evaluation environment
└── pyproject.toml
```

Local `paper/`, `benchmark-results/`, `Evaluation/`, and `external/` directories are intentionally excluded from Git by [`.gitignore`](.gitignore). The PNGs in [`docs/readme-assets/`](docs/readme-assets/) are rendered from the supplied vector figures in `paper/image/` so that the README images display in a source checkout. Their [source mapping](docs/readme-assets/README.md) is recorded alongside the assets.

---

## 🚀 Quick Start

### 1. Install the core package

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[dev]"
```

### 2. Run a complete local example

```bash
python examples/end_to_end.py
sepm --db sepm_demo.db candidates
sepm --db sepm_demo.db retrieve "verified catalog results"
```

The example registers an agent, creates workspaces, proposes and validates a procedure, records a successful trial, publishes a version, and retrieves it. It writes `sepm_demo.db` in the current directory.

### 3. Check the implementation

```bash
pytest -q
sepm-benchmark --output benchmark-results/mechanism/report.json
```

`sepm-benchmark` exercises deterministic divergence, evidence, promotion, and retrieval checks without an LLM service.

### Integrate with an agent system

`SEPMService` is the Python facade. [`AgentMemoryAdapter`](src/sepm/adapter.py) exposes agent-start, agent-step, and SOP-outcome hooks for host frameworks. The memory layer does not require changing the host's agent count or communication topology.

```python
from sepm.adapter import AgentMemoryAdapter
from sepm.service import SEPMService

adapter = AgentMemoryAdapter(SEPMService("sepm.db"))
context = adapter.on_agent_start(profile, workspace, task_query, query_plan)
step_result = adapter.on_agent_step(blackboard_entry)
adapter.on_sop_outcome(sop_id, success=True, query=task_query)
```

See [`examples/end_to_end.py`](examples/end_to_end.py) for concrete model objects and a runnable procedure lifecycle.

---

## 🧪 Evaluation

The checked-in [`evaluation_matrix.json`](evaluation_matrix.json), [`manifests/`](manifests/), and [`scripts/run_paper_experiments.sh`](scripts/run_paper_experiments.sh) define the **local selected-case evaluation workflow** and its adapters. Full runs need the upstream benchmark repositories under `external/`, benchmark-specific services, and model credentials. The current selected-case matrix and ignored local result files are **not the complete 134/812/300-task artifact set behind the manuscript's headline table**.

For the full dependency environment and a configured server:

```bash
conda env create -f environment.yml
conda activate sepm
cp .env.example .env          # Then fill in your own credentials and endpoints
python scripts/validate_paper_setup.py --require-jobs
bash scripts/run_paper_experiments.sh all
```

The runner has `preflight`, `smoke`, `matrices`, `generality`, `sop-models`, `ablation`, and `report` stages. It writes resumable results under `benchmark-results/`; [`scripts/validate_paper_setup.py`](scripts/validate_paper_setup.py) checks required jobs and external paths before a formal run. Use `python -m sepm.evaluation_runner --help` and `python -m sepm.ablation.runner --help` for matrix filters and individual stages.

---

## 📄 Paper and License

**Manuscript:** *SEPM: Self-Evolving Procedural Memory Grounded in Task Graphs for Multi-Agent Collaboration* (anonymous ICLR 2027 submission). The draft PDF is in `paper/SEPM.pdf` in this workspace; `paper/` is not part of the source release. Please use the public manuscript's bibliographic record when one becomes available.

This code is licensed under [Apache 2.0](LICENSE).
