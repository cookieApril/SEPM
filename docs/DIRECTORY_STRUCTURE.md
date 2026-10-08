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
├── scripts/
├── src/
│   └── sepm/
├── tests/
└── adapters/
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
| `src/sepm/models.py` | Pydantic domain models for agents, blackboard entries, evidence, SOP candidates, SOP versions, and retrieval results. |
| `src/sepm/service.py` | Main facade for private memory, blackboard writes, divergence detection, state resolution, SOP validation, promotion, and retrieval. |
| `src/sepm/storage.py` | SQLite schema, transactions, candidate leases, immutable SOP versions, CAS updates, and audit log. |
| `src/sepm/safety.py` | Deterministic safety validation for SOP create/update/delete candidates. |
| `src/sepm/divergence.py` | Goal, plan, and world-state divergence reporting. |
| `src/sepm/evidence.py` | Evidence-priority state resolution. |
| `src/sepm/retrieval.py` | BM25 plus vector SOP retrieval. |
| `src/sepm/config.py` | Immutable runtime configuration and component-condition switches. |
| `src/sepm/adapter.py` | Framework-independent lifecycle adapter. |
| `src/sepm/ablation/` | Minimal component-condition mapping and runner. |
| `src/sepm/benchmark.py` | Deterministic mechanism diagnostics. |
| `src/sepm/evaluation_runner.py` | Matrix expansion, external job orchestration, and report aggregation. |
| `src/sepm/paper_tables.py` | Table generation for cross-benchmark results, SOP-model sensitivity, and minimal ablation. |

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
SEPMService
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
ignores the entire tree, including plans, checkpoints, SQLite files, logs, raw
trajectories, snapshots, result tables, and figures. Manuscript files and
visualization code under `paper/` are also local-only. Third-party source trees
under `external/` remain local-only as well.
