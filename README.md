# Team Memory Framework

Team Memory is a pure-Python memory layer for multi-agent systems. It separates
three kinds of memory:

- `Private Memory`: persistent role information and the current task;
- `Shared Blackboard`: lightweight structured execution records, canonical
  goals, task dependency graphs, task state, and evidence;
- `Procedural Memory`: versioned SOPs represented as procedure, metadata, and
  procedure graph, with general SOPs separated from context-specific variants.

The core principle is:

```text
successful trajectory != promotable procedural memory
```

Maintained documentation:

- [docs/DIRECTORY_STRUCTURE.md](docs/DIRECTORY_STRUCTURE.md)
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- [docs/RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md)
- [docs/STORAGE_SCHEMA.md](docs/STORAGE_SCHEMA.md)

## Implemented Mechanisms

1. `Goal Divergence`: embedding distance plus a replaceable judge.
2. `Plan Divergence`: graph-edit proxy over action nodes and dependency edges.
3. `State Divergence`: conflict detection over shared state keys.
4. Evidence resolution: `Authoritative State > Tool Observation > Verified Artifact > Agent Inference`.
5. Alignment workflow: `Detect -> Classify -> Verify -> Resolve -> Realign`.
6. Safety-aware SOP update gate: utility is estimated as `Success - lambda * Cost`,
   then deterministic safety validation rejects risky shortcuts before a shared
   write.
7. SOP promotion gates: safety, state verification, causal support, first qualified validation trial, and positive net benefit.
8. Atomic version updates: SQLite transaction plus `base_version` compare-and-swap; old versions remain auditable.
9. SOP retrieval: `Candidate(q)=BM25(q) union VectorSearch(q)`, then lexical and semantic relevance ranking.
10. Stateless SOP workers: candidates enter a persistent queue; validator workers claim leases and can be horizontally scaled.

## Installation

For a lightweight development installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

For the full evaluation environment, create or update the Conda environment:

```bash
conda env create -f environment.yml
conda env update -f environment.yml --prune
conda activate team-memory
python -m pip show team-memory-plugin
```

The full environment includes optional dependencies for the external GMemory
and WebArena adapters. External benchmark repositories are installed separately
under `external/` and are intentionally not vendored in this repository.

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/team_memory/` | Installable Team Memory implementation and evaluation runner. |
| `adapters/` | Benchmark-facing adapters for GMemory and cross-benchmark runs. |
| `tests/` | Unit and integration-contract tests that do not require model calls. |
| `scripts/` | Reproducible experiment, smoke-test, and result-building entry points. |
| `manifests/` | Frozen case manifests used by the paper experiments. |
| `paper/code/` | Figure/table generation source and compact derived data. |
| `benchmark-results/` | Local generated outputs plus one checked-in development snapshot. |
| `docs/` | Method, storage, directory, and experiment specifications. |

Raw traces, logs, checkpoints, databases, third-party repositories, manuscript
drafts, and rendered files are excluded from Git. See
[`benchmark-results/README.md`](benchmark-results/README.md) for the artifact
policy.

## Basic Usage

```python
from team_memory import TeamMemoryService

memory = TeamMemoryService("team_memory.db")
```

Read-only CLI checks:

```bash
team-memory --db team_memory.db candidates
team-memory --db team_memory.db retrieve "verified catalog ranking"
```

## Agent Integration

`TeamMemoryService` is the full Python facade. `AgentMemoryAdapter` provides
framework-independent lifecycle hooks:

```python
from team_memory.adapter import AgentMemoryAdapter
from team_memory.service import TeamMemoryService

adapter = AgentMemoryAdapter(TeamMemoryService("experiment.db"))
context = adapter.on_agent_start(profile, workspace, task_query, query_plan)
step_result = adapter.on_agent_step(blackboard_entry)
adapter.on_sop_outcome(sop_id, success=True, query=task_query)
```

Team Memory does not add a new Manager-Agent and does not change the native
number of agents or communication topology in AutoGen, DyLAN, MacNet, or MARBLE.
If a framework already has a coordinator, retrieval and divergence signals can
be injected into that native role.

## Core API

| Method | Purpose | Writes |
| --- | --- | ---: |
| `register_agent` | Store private memory. | Yes |
| `create_workspace` | Store canonical task goal and task plan. | Yes |
| `append_blackboard` | Append compact task/state/evidence record. | Yes |
| `detect_divergence` | Compute goal, plan, and state divergence. | No |
| `resolve_state` | Resolve conflicts by evidence hierarchy. | No |
| `propose_sop` | Safety precheck and enqueue candidate SOP. | Yes |
| `validate_candidate` | Check candidate state and causal evidence. | Yes |
| `record_reproduction` | Record validation trial and publish after the first qualified success. | Yes |
| `promote_candidate` | Idempotently publish an eligible SOP. | Yes |
| `retrieve_sops` | Recall SOPs with BM25 + VectorSearch and rank by lexical/semantic relevance. | No |
| `record_sop_outcome` | Record whether a retrieved SOP helped a later task. | Yes |

## SOP Promotion

A candidate can be published after one qualified successful trial. The trial must
have state verification, causal support, and positive net benefit:

```text
candidate_reward - baseline_reward - cost > 0
```

Candidate-level update value is recorded as `Success - lambda * Cost`; the
validation trial then supplies the measured net benefit above. Safety validation
is not a reward feature. It rejects updates whose risk exceeds the configured
threshold or that remove mandatory confirmation, backup, authorization, or
irreversible-operation protection. Deletion is logical deactivation, not history
removal. Sensitive or critical SOPs cannot be deleted outright; they must be
safely superseded by a newer version.

## Retrieval

Retrieval is for reuse: it ranks SOPs that are likely to help the current task.
The default ranking uses:

- BM25 lexical relevance over SOP title, metadata, applicability, context
  conditions, and step text;
- vector semantic relevance over the same SOP document.

Safety is not a retrieval weight. Mandatory safety steps are protected by the
SOP update/delete/supersede validator. Environment changes should be handled by
failed outcomes, verified state evidence, and newer SOP versions.

## Evaluation

The paper experiment design keeps three result tables: cross-MAS generality,
SOP-model sensitivity, and the minimal component ablation. The current ablation
configuration defines `no-extra-components`, `blackboard-only`, `sop-only`,
`divergence-only`, and `full`.
Mechanism diagnostics remain engineering tests, not paper experiments. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the complete protocol.

Evaluation resume is cell-level: each benchmark/task/method/MAS/model/seed/case
and condition has its own result, log, and SQLite checkpoint row when a case id
is selected. Re-running the matrix skips successful cells and retries only
incomplete or explicitly selected failed cells.

Run the deterministic, model-free checks with:

```bash
team-memory-benchmark --output benchmark-results/mechanism/report.json
pytest -q
```

Full benchmark runs require the external repositories and API credentials
described in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Copy `.env.example` to
`.env`; never commit the populated file.
