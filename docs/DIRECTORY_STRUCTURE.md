# Directory Structure

This repository keeps only four documentation files under `docs/`:

| File | Purpose |
|---|---|
| `docs/DIRECTORY_STRUCTURE.md` | Directory map and file responsibilities. |
| `docs/EXPERIMENTS.md` | Full experiment design with the component ablation reduced to four conditions. |
| `docs/RESEARCH_DESIGN.md` | Scientific contract mapping implementation to method. |
| `docs/STORAGE_SCHEMA.md` | Storage and graph schema for blackboard, task graphs, and SOP versions. |

## Repository Map

```text
Multi-Agent Memory/
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── environment.yml
├── evaluation.xml
├── evaluation_matrix.json
├── pyproject.toml
├── docs/
│   ├── DIRECTORY_STRUCTURE.md
│   ├── EXPERIMENTS.md
│   ├── RESEARCH_DESIGN.md
│   └── STORAGE_SCHEMA.md
├── examples/
├── manifests/
├── paper/
│   └── code/
├── scripts/
├── src/
│   └── team_memory/
├── tests/
├── adapters/
└── benchmark-results/
    ├── README.md
    └── smoke/gmemory-snapshots/multiagentbench/
```

## Root Files

| File | Responsibility |
|---|---|
| `README.md` | Project overview, memory layers, core API, and pointers to the four maintained docs. |
| `environment.yml` | Conda environment definition. |
| `evaluation.xml` | Short evaluator facts for core safety, evidence, concurrency, and retrieval rules. |
| `evaluation_matrix.json` | Structured experiment matrix and five component conditions. |
| `pyproject.toml` | Package metadata, dependencies, and command entry points. |

## Source Package

| Path | Responsibility |
|---|---|
| `src/team_memory/models.py` | Pydantic domain models for agents, blackboard entries, evidence, SOP candidates, SOP versions, and retrieval results. |
| `src/team_memory/service.py` | Main facade for private memory, blackboard writes, divergence detection, state resolution, SOP validation, promotion, and retrieval. |
| `src/team_memory/storage.py` | SQLite schema, transactions, candidate leases, immutable SOP versions, CAS updates, and audit log. |
| `src/team_memory/safety.py` | Deterministic safety validation for SOP create/update/delete candidates. |
| `src/team_memory/divergence.py` | Goal, plan, and world-state divergence reporting. |
| `src/team_memory/evidence.py` | Evidence-priority state resolution. |
| `src/team_memory/retrieval.py` | BM25 plus vector SOP retrieval. |
| `src/team_memory/config.py` | Immutable runtime configuration and component-condition switches. |
| `src/team_memory/adapter.py` | Framework-independent lifecycle adapter. |
| `src/team_memory/ablation/` | Minimal component-condition mapping and runner. |
| `src/team_memory/benchmark.py` | Deterministic mechanism diagnostics. |
| `src/team_memory/evaluation_runner.py` | Matrix expansion, external job orchestration, and report aggregation. |
| `src/team_memory/paper_tables.py` | Table generation for cross-benchmark results, SOP-model sensitivity, and minimal ablation. |

## Tests

| File | Responsibility |
|---|---|
| `tests/test_core.py` | Core service invariants: evidence resolution, safety, divergence, promotion, retrieval, and concurrency. |
| `tests/test_ablation.py` | Five component conditions: no added mechanism, blackboard-only, SOP-only, divergence-only, and full. |
| `tests/test_benchmark.py` | Mechanism diagnostic report shape and deterministic gold cases. |
| `tests/test_evaluation_runner.py` | Matrix expansion, checkpointing, provider-key isolation, and external adapter contract. |
| `tests/test_paper_tables.py` | Table-generation invariants for full experiments and minimal component ablation. |
| `tests/test_validate_setup.py` | Setup validation for configured external jobs. |

## Core Call Chain

```text
Agent framework / adapter / benchmark
       |
       v
TeamMemoryService
 ├─ DivergenceDetector + EvidenceResolver
 ├─ SafetyValidator
 ├─ SOPRetriever
 └─ SQLiteStore
       ├─ blackboard
       ├─ candidates + trials
       ├─ sop_versions + sop_heads
       └─ audit_log
```

SOP writes follow:

```text
propose_sop -> validate_candidate -> record_reproduction -> promote_candidate
```

Promotion requires utility, safety, state verification, causal support, a
qualified validation trial, positive net benefit, and an atomic version commit.

## Generated Artifacts

`benchmark-results/` is created and populated by evaluation commands. Git
ignores its plans, checkpoints, SQLite files, logs, raw trajectories, and result
tables. The only versioned files in that tree are its policy README and the
small frozen MultiAgentBench development snapshot required by integration
tests. Third-party source trees under `external/` are also local-only.
