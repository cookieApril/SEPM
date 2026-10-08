# SEPM Experiment Design

This document is the canonical experiment design for the current contribution
set. The paper-facing package contains only three experiment tables:

```text
E1 Cross-Benchmark Generality
E2 SOP-Model Sensitivity
E3 Minimal Component Ablation
```

The next component-ablation protocol has exactly five conditions:

```text
no-extra-components, blackboard-only, sop-only, divergence-only, full
```

Operational server commands, temporary logs, smoke-test transcripts, and
runbook details do not belong in this file.

## Runtime Support Status

Manifest-only adapter checks are engineering smoke tests, not paper evidence.
They are allowed only when `SEPM_ALLOW_MANIFEST_SMOKE=1` is set. Formal
cells must use an official runtime harness and must report the official
single-case score with `case_count = 1.0`.

The target experiment protocol is broader than the current implementation
status. Missing cells must be implemented before running paper experiments; they
must not be silently dropped from the design and must not be replaced by
manifest-only or benchmark-native smoke scores.

The integration contract is now `sepm.benchmark_runtime`: official
benchmark loops should call its `start_agent`, `observe`, `before_action`,
`after_action`, `record_error`, and `finish_metrics` hooks so SEPM is
actually connected to observations, actions, results, errors, SOP retrieval,
blackboard writes, divergence detection, and final result metrics.

Current implementation status:

| Area | Status | Required before paper run |
|---|---|---|
| ALFWorld | Existing adapter reuses the official G-Memory environment and MAS runner for `no-memory`, `gmemory`, and `sepm`. | Add a native SEPM/AgentNet bridge if AgentNet is included for ALFWorld. |
| WebArena | Official single-case runtime can start. | Inject `SEPMBenchmarkRuntime` into the web agent loop for AutoGen/AgentNet/DyLAN and memory baselines. |
| MultiAgentBench/MARBLE | Official single-case config can start. | Inject runtime hooks into MARBLE agent message/action loop and propagate SEPM metrics. |
| OfficeBench | Official single-case runtime can start after Docker permissions are fixed. | Inject runtime hooks into the office agent interaction loop and propagate SEPM metrics. |

## Core Contributions

SEPM is a procedural-memory-oriented layer for existing multi-agent
systems. It does not add a new task-solving agent and does not change the native
topology of AutoGen, AgentNet, DyLAN, or another host MAS.

The paper validates three contributions:

1. Safety-aware adaptive procedural memory.
   The system retrieves, creates, updates, specializes, deletes, and atomically
   commits reusable SOPs. Candidate updates are evaluated by success, cost, and
   safety. Unsafe updates that remove confirmation, authorization, backup, or
   other mandatory constraints must be rejected.

2. Procedural-memory-oriented lightweight shared blackboard.
   Agents write only structured execution records:

   ```text
   {task, observation, action, result/outcome, error}
   ```

   Each workspace also maintains a task dependency graph. The event stream and
   task graph should be sufficient for later SOP extraction, warning extraction,
   and procedure-graph construction without storing full reasoning traces.

3. Divergence-aware multi-agent state alignment.
   The manager/coordinator detects and resolves goal, plan, and world-state
   divergence using the pipeline:

   ```text
   Detect -> Classify -> Verify -> Resolve -> Realign
   ```

   World-state conflicts are resolved by evidence priority before agent
   negotiation:

   ```text
   Authoritative State > Tool Observation > Verified Artifact > Agent Inference
   ```

## Host-Agent Memory Setting

SEPM is not an individual-agent long-term memory method. The durable
experience is stored in shared procedural memory, while each host MAS keeps its
native execution state:

| MAS | Native agent memory setting in experiments | Paper note |
|---|---|---|
| AutoGen | Short-term conversation/context memory only. | No durable per-agent memory beyond the current run. |
| DyLAN | Short-term conversation/context memory only. | Dynamic debate state is local to the run. |
| AgentNet | AgentNet's built-in agent memory remains enabled. | Treat AgentNet memory as part of the host MAS, not as SEPM. |

This distinction must be stated in the paper. Comparisons are therefore between
host-MAS memory settings plus optional external memory layers, not between
identical stateless agents in every MAS.

## Evidence Tracks

| Track | Purpose | Paper role |
|---|---|---|
| Cross-Benchmark generality | Test whether SEPM improves modern long-horizon agent tasks when added to existing MAS frameworks. | Main evidence. |
| SOP-model sensitivity | Test whether results depend on one SOP extraction/validation model. | Robustness evidence. |
| Minimal component ablation | Test whether SOP evolution and divergence alignment each contribute. | Mechanism evidence. |

`SEPMBench` and other deterministic checks are engineering diagnostics.
They may remain in unit tests, but they are not paper experiments.

## Claim-Evidence Map

| Claim | Experiment | Required comparison | Main metrics |
|---|---|---|---|
| SEPM improves task performance without changing MAS topology. | E1 Cross-Benchmark generality | Same benchmark, MAS, actor, deterministic run configuration, and case; compare No-memory, selected memory baselines, and SEPM. | Official benchmark score and macro average. |
| SOP extraction results do not depend on a single SOP model. | E2 SOP-model sensitivity | Fixed AutoGen, actor, cases, deterministic run configuration, prompts, and budget; vary only SOP model. | Task score, JSON validity, candidate pass rate, evidence correctness, safety rejection, cost. |
| SOP evolution and divergence alignment are both useful additions. | E3 Minimal ablation | Four matched conditions only under AutoGen. | Task score, SOP reuse, divergence recovery, unsafe accepted, token overhead, latency. |

## E1 Cross-Benchmark Generality

Purpose: test whether SEPM works as a plug-in procedural-memory layer
across modern task families and host MAS topologies.

Benchmarks:

- ALFWorld: procedural embodied/text-world anchor.
- WebArena: web interaction with persistent state, verification, and
  authorization-sensitive actions.
- MultiAgentBench: native multi-agent collaboration/coordination tasks.
- OfficeBench: long office workflows over documents, spreadsheets, email, and
  related productivity artifacts.

Fixed:

- actor model;
- SOP model for SEPM;
- benchmark subset;
- prompt and token budget;
- matched case set and deterministic run configuration within each benchmark;
- native MAS topology.

All methods/conditions are evaluated on the same matched case IDs under a fixed
deterministic run configuration. `seed=0` is kept only as an internal
compatibility field for runner/checkpoint/result identity, not as a paper
experiment dimension.

Target E1 matrix:

| Benchmark family | MAS/framework cells | Memory-method cells | Paper use |
|---|---|---|---|
| ALFWorld | AutoGen, AgentNet, DyLAN | No-memory, selected baselines, SEPM | Procedural-memory anchor. |
| WebArena | AutoGen, AgentNet, DyLAN | No-memory, selected baselines, SEPM | Web state, authorization, and tool verification. |
| MultiAgentBench/MARBLE | AutoGen, AgentNet, DyLAN | No-memory, selected baselines, SEPM | Multi-agent coordination and divergence. |
| OfficeBench | AutoGen, AgentNet, DyLAN | No-memory, selected baselines, SEPM | Long office workflows and artifact state. |

Current implementation status is tracked separately from the target matrix.
Cells that do not yet inject the host MAS and memory method into the official
single-case runtime must remain blocked, not reported.

The full E1 expansion request is tracked in
`benchmark-results/unified-v3/tables/e1_full_completion_gap_audit.md`, with the
frozen full MultiAgentBench manifest at
`manifests/e1_multiagentbench_full_manifest.json`. That manifest contains all
400 official MultiAgentBench records: 100 Research, 100 Database, 100 Coding,
and 100 Minecraft. Under the requested seed-0 AutoGen/DyLAN x no-memory,
G-Memory, and SEPM matrix, the expanded E1 target has 2760 cells:
72 ALFWorld cells, 144 WebArena cells, 144 OfficeBench cells, and 2400
MultiAgentBench cells. The current accepted 168-cell subset remains preserved,
but the remaining 2592 cells are gated on official single-case smoke evidence.
The implementation now exposes a benchmark-neutral G-Memory bridge over frozen
development snapshots and a host bridge for AutoGen/DyLAN identity, topology
hashing, routing traces, and actor-visible context injection in the existing
WebArena, OfficeBench, and MARBLE runtime hooks. The first new smoke path
(`webarena/autogen/gmemory/219`) invoked the G-Memory bridge and wrote retrieval
evidence, but the official WebArena browser loop could not reach the map service
at `http://18.117.138.7:3000/`, so no official score was produced and the
expanded sweep remains blocked. Metadata-only host or memory labels must not be
reported as completed E1 evidence.

Current E1 runnable-scope audit is stored at
`benchmark-results/unified-v3/tables/e1_readiness_report.md`, and the paired
paper-runnable plan is stored at
`benchmark-results/unified-v3/tables/e1_paper_runnable_plan.md`. The runnable
plan contains 168 cells: ALFWorld AutoGen/DyLAN with `no-memory`, `gmemory`, and
`sepm`; WebArena AutoGen with `no-memory` and `sepm`; and
OfficeBench AutoGen with `no-memory` and `sepm`. MultiAgentBench/MARBLE
is excluded from the E1 main plan until a Research-only paired plan is
explicitly adopted and smoke-tested. AgentNet is excluded because matched
baselines are missing, and `agent-native-memory`, `generative-memory`, and
`mem0` are excluded because they do not yet have real official-loop injection.
Smoke anchors may use known official cases to validate wiring, but they are not
paper E1 results unless they are part of the final matched subset. ALFWorld E1
uses the official GMemory adapter path for `no-memory`, `gmemory`, and
`sepm`; that adapter imports Chroma-backed memory backends at module load
time, so the runnable environment includes `langchain==0.3.25`,
`langchain-chroma==0.2.3`, `finch_clust==0.2.0`, and `finchpy==0.0.1`. The
GMemory delegate also applies the same ALFWorld/TextWorld `EvalSymbol`
compatibility patch used by the SEPM ALFWorld loop before constructing
the official environment; this preserves the official task/evaluator path while
avoiding Python runtime locals handling failures during reset. WebArena E1 uses
the official `run.py` and evaluator unchanged; its legacy OpenAI chat provider
has a runtime compatibility patch for `gpt-5*` models that omits unsupported
sampling parameters and uses the compatible completion-token parameter. The
WebArena official runtime also requires Playwright and a local Chromium browser
bundle; the current Python 3.13 runner uses `playwright==1.62.0` with Chromium
installed through `python -m playwright install chromium`. Because WebArena's
import-time `beartype` decorators reject NumPy 2.x typing aliases under Python
3.13, the runtime entry disables those type-check decorators in this environment
without changing evaluator scoring. HuggingFace-only provider imports are lazy
loaded so the OpenAI E1 path does not require unused HuggingFace runtime
dependencies. The WebArena runtime shim also adapts the legacy OpenAI exception
handler to OpenAI SDK 2.x and uses a longer `domcontentloaded` navigation wait
for SEPM subprocess startup. WebArena observation screenshots use
`WEBARENA_PLAYWRIGHT_TIMEOUT_MS=90000` by default to avoid Python 3.13/Chromium
load-event timeouts on official map tasks. The same environment-controlled
timeout is used for evaluator navigation, without changing evaluator scoring
predicates or PASS/FAIL logic.

The new-path smoke gate report is stored at
`benchmark-results/unified-v3/tables/e1_new_path_smoke_gate_report.md`. It is an
engineering gate only and is not paper evidence.

Reporting:

- report one block per genuinely wired MAS/framework;
- do not average across MAS as if frameworks were random seeds;
- report official benchmark metrics, then a macro average only across matched
  comparable cells;
- for AgentNet, explicitly state that its built-in agent memory is part of the
  host MAS baseline once AgentNet cells are truly wired.

MultiAgentBench/MARBLE scoring:

- `primary_score` must come from official MARBLE evaluator artifacts, never
  from SEPM sidecar metrics.
- Database tasks use MARBLE's `task_evaluation` payload and the official
  `scripts/database/batch_eval.py` root-cause label matching procedure. The
  adapter records this source as `task_evaluation.database_batch_eval`.
- Research/world-style tasks use numeric `task_evaluation` ratings emitted by
  `marble/evaluator/evaluator.py`.
- Coding tasks use `code_quality` only when the official loop generated the
  required workspace solution artifact. If a coding run finishes without
  `code_quality`, `task_evaluation`, `score`, or `final_score`, the adapter
  must refuse to write a paper result.
- `planning_scores`, `communication_scores`, `total_milestones`, and
  `agent_kpis` are collaboration diagnostics. They may be reported as
  secondary metadata, but they are not the task `primary_score`.

## E2 SOP-Model Sensitivity

Purpose: separate task execution ability from SOP extraction and validation
quality.

Fixed:

- MAS: AutoGen;
- actor model: `gpt-5-mini`;
- memory method: SEPM;
- benchmark subset, deterministic run configuration, prompt budget, adapter protocol.

Vary only SOP model.

Configured SOP models:

| Group | SOP models |
|---|---|
| Closed-source | `gpt-5.6-terra`, `claude-opus-5` |
| Open-source relay models | `qwen3.8-max`, `deepseek-v4-flash-0731`, `kimi-k3`, `glm-5.2` |
| Local small models | `gemma-4-31b-it`, `gemma-4-12b-it`, `qwen3.5-27b`, `qwen3.5-9b`, `qwen3.5-2b`, `qwen3.5-0.8b` |
 
Before large parallel runs, first validate at least two locally deployed small
models end to end on the same subset. Only after their endpoint, JSON parsing,
cost accounting, and resume behavior pass should the full local-model sweep be
parallelized.

Benchmarks and subset rationale:

| Benchmark | Use in E2 | Reason |
|---|---|---|
| ALFWorld | Matched procedural subset. | Tests whether SOP extraction handles action sequences and reusable procedures. |
| WebArena | Matched web-task subset. | Tests whether SOP validation preserves state checks, authorization, and tool-observation grounding. |
| MultiAgentBench Research | Matched collaboration subset with stable official `task_evaluation` scoring. | Tests whether SOP extraction remains stable when tasks require role coordination and dependency tracking. |
| OfficeBench | Matched office-workflow subset. | Tests whether SOP extraction generalizes to long workflows with documents, sheets, and cross-artifact state. |

Use a subset rather than a full sweep: E2 is a curator-model robustness test, not
the main performance table. The subset must be identical for every SOP model.

E2 fixed subset size and selection rule:

| Benchmark | Cases | Selection rule |
|---|---:|---|
| ALFWorld | 12 | Two official cases from each procedural family: pick-and-place, clean-then-place, heat-then-place, cool-then-place, examine/look, and pick-two-object. |
| WebArena | 24 | Four cases from each web-operation bucket: information lookup, navigation/filtering, form or configuration update, content creation/editing, account/cart/order state, and developer/CMS operation. |
| MultiAgentBench Research | 12 | Research-focused deterministic subset matching the current MARBLE official scoring path. Database and Coding remain blocked at the environment/scoring layer and are not paper results; Minecraft is excluded from E2 until the same official scoring path is verified. |
| OfficeBench | 24 | Eight Single-App, eight Two-App, and eight Three-App cases, balanced across exact, fuzzy, and execution-based evaluators when available. |

The final committed manifest must contain concrete official task IDs. Until the
official repositories are downloaded, the above table fixes the subset contract
but not the repository-specific IDs.

Case-ID manifest selection policy:

1. Use only official benchmark task IDs from the downloaded benchmark manifests.
2. Assign every official task to one of the strata listed above using official
   metadata first. If metadata is missing, use the task path, environment name,
   evaluator type, website/app label, or task instruction pattern, and record the
   rule in the manifest note.
3. Within each stratum, sort candidate IDs by:

   ```text
   sha256(bytes.fromhex("7465616d2d6d656d6f72792d65322d7631")
          + benchmark_id.encode() + official_case_id.encode())
   ```

   Select the first `n` IDs required by the table. Do not use dataset order,
   model performance, failed pilot runs, or manual convenience to choose IDs.
4. If a stratum has fewer than the required cases, take all available cases and
   redistribute the deficit to the closest stratum inside the same benchmark,
   using the same hash order.
5. Commit the final IDs to `evaluation_matrix.json.case_ids`. The same IDs must
   be reused for every SOP model and retry.

Report:

- official task score;
- SOP JSON validity;
- candidate pass rate;
- evidence-reference correctness;
- safety rejection count;
- SOP-model token/cost.

## E3 Minimal Component Ablation

Purpose: isolate the two added mechanisms without redundant micro-ablations.

Fixed:

- MAS: AutoGen;
- actor model: `gpt-5-mini`;
- SOP model: `gpt-5-mini`;
- same matched cases, deterministic run configuration, prompts, and budget across the five conditions.

Use exactly five conditions:

| Condition | Procedural SOP | Blackboard | Divergence alignment | Purpose |
|---|---:|---:|---:|---|
| `no-extra-components` | Off | Off | Off | Same MAS and actor model with no added SEPM mechanism. |
| `blackboard-only` | Off | On | Off | Isolate the structured shared execution record substrate. |
| `sop-only` | On | On | Off | Test procedural memory plus the required blackboard substrate. |
| `divergence-only` | Off | On | On | Test state alignment plus the required blackboard substrate. |
| `full` | On | On | On | Test the complete SEPM configuration. |

Benchmarks and subset rationale:

| Benchmark | Priority | Reason |
|---|---|---|
| MultiAgentBench | Primary | Most directly exercises goal, plan, and state divergence between agents. |
| OfficeBench | Primary | Long workflows expose task dependencies, artifact state, and error-to-warning conversion. |
| WebArena | Primary | Web state and authorization-sensitive operations stress safety-aware SOP validation. |
| ALFWorld | Small anchor subset | Keeps a classic procedural-memory anchor without turning E3 into a full generality rerun. |

E3 fixed subset size and selection rule:

| Benchmark | Cases | Selection rule |
|---|---:|---|
| MultiAgentBench | 12 | Research-focused deterministic subset. Research tasks currently provide stable official `task_evaluation` scoring; Database remains blocked at the official Docker environment layer, and Coding remains blocked unless the official loop emits the required solution artifact and `code_quality`. Blocked Database/Coding runs are not paper results. |
| OfficeBench | 18 | Six Two-App and twelve Three-App cases, prioritizing cross-artifact state, dependent operations, and execution-based evaluation. |
| WebArena | 18 | Six account/cart/order cases, six GitLab/CMS or admin update cases, and six content/forum/wiki mutation cases. Pure information lookup is excluded from E3 because it rarely exercises safety-aware update or state realignment. |
| ALFWorld | 12 | Two official cases from each procedural family, matching the E2 ALFWorld anchor. |

All five ablation conditions must use exactly the same case IDs within each
benchmark.

ALFWorld E3 smoke/procedural-anchor set:

- The first three ALFWorld cases in `manifests/e3_case_manifest.json` are marked
  as the smoke/procedural-anchor set. They are used to verify the E3 result
  schema and ablation switches before running broader E3 cells.
- This smoke set does not represent the full E3 matched subset. The full ALFWorld
  E3 subset remains the 12-case deterministic matched subset listed in
  `evaluation_matrix.json` and `manifests/e3_case_manifest.json`.

E3 case IDs use the same deterministic manifest policy as E2, except the frozen
hash seed is:

```text
7465616d2d6d656d6f72792d65332d7631
```

The seeds are stored as hexadecimal bytes to preserve the original case ordering
across the project rename. Changing these bytes would define a new subset.

The E3 manifest must not be selected from observed ablation performance. If an
official task is removed because it cannot run under the adapter, document the
infrastructure reason and replace it by the next hash-ranked ID from the same
stratum.

Do not add paper-level ablations for:

- w/o shared blackboard;
- w/o evidence hierarchy;
- w/o safety gate;
- w/o causal gate;
- w/o goal divergence only;
- w/o plan divergence only;
- w/o state divergence only;
- fixed retrieval;
- one validator rule removed.

Those switches may be retained as internal tests, but they should not appear in
the component ablation table.

Metrics:

- official task score;
- paired delta against `no-extra-components`;
- SOP reuse success;
- divergence recovery;
- unsafe accepted;
- token overhead;
- latency.

All methods/conditions are evaluated on the same matched case IDs under a fixed
deterministic run configuration. Cases denotes the number of benchmark
instances. We report the exact case counts in Appendix X. `seed=0` is kept only
as an internal compatibility field for runner/checkpoint/result identity, not as
a paper experiment dimension.

## Result Artifacts

Result data, derived tables, plots, manuscript sources, and visualization code
are intentionally excluded from the public repository. Evaluation commands write
local artifacts under `benchmark-results/`, which is ignored by Git. The tracked
case manifests and experiment protocols remain sufficient to configure the runs.

## Execution and Resume Policy

The fair experimental unit is:

```text
experiment_id x benchmark/task x memory_method x mas_framework
x actor_model x sop_model x case_id x condition
```

Every paper benchmark must provide a matched `case_ids` manifest before formal
runs. Each case is checkpointed separately. `seed=0` is kept only as an internal
compatibility field for runner/checkpoint/result identity, not as a paper
experiment dimension. The minimal paper resume unit is:

```text
experiment_id x benchmark/task x memory_method x mas_framework
x actor_model x sop_model x case_id x condition
```

A resumed run skips successful cells and retries only incomplete or explicitly
selected failed cells. Legacy results may be imported only when the full cell
identity matches the current matrix.

## Evidence Boundary

- Generated scores, completion audits, tables, and figures remain local.
- Smoke and preflight checks are engineering checks, not paper evidence.
- Runtime/no-score runs, failed runs, sidecars, and partial rows are not accepted
  paper results.
- Compatibility adaptations must not change official evaluator scoring logic.
- WebArena infrastructure state must be controlled by the run plan; audits must
  not trigger deferred cells or treat them as scored failures.

## Evidence Rules

- Do not invent result values; use `TBD` until full runs are available.
- Do not report smoke tests as paper evidence.
- Do not remove E1 or E2 when simplifying E3.
- Do not add ablation rows beyond the five-condition protocol.
- Do not merge unrelated benchmark families into one unsupported headline
  score.
- Keep actor model, MAS topology, benchmark case, deterministic run
  configuration, and prompt fixed inside
  paired comparisons.
