# Team Memory Experiment Design

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
They are allowed only when `TEAM_MEMORY_ALLOW_MANIFEST_SMOKE=1` is set. Formal
cells must use an official runtime harness and must report the official
single-case score with `case_count = 1.0`.

The target experiment protocol is broader than the current implementation
status. Missing cells must be implemented before running paper experiments; they
must not be silently dropped from the design and must not be replaced by
manifest-only or benchmark-native smoke scores.

The integration contract is now `team_memory.benchmark_runtime`: official
benchmark loops should call its `start_agent`, `observe`, `before_action`,
`after_action`, `record_error`, and `finish_metrics` hooks so Team Memory is
actually connected to observations, actions, results, errors, SOP retrieval,
blackboard writes, divergence detection, and final result metrics.

Current implementation status:

| Area | Status | Required before paper run |
|---|---|---|
| ALFWorld | Existing adapter reuses the official G-Memory environment and MAS runner for `no-memory`, `gmemory`, and `team-memory`. | Add a native Team Memory/AgentNet bridge if AgentNet is included for ALFWorld. |
| WebArena | Official single-case runtime can start. | Inject `TeamMemoryBenchmarkRuntime` into the web agent loop for AutoGen/AgentNet/DyLAN and memory baselines. |
| MultiAgentBench/MARBLE | Official single-case config can start. | Inject runtime hooks into MARBLE agent message/action loop and propagate Team Memory metrics. |
| OfficeBench | Official single-case runtime can start after Docker permissions are fixed. | Inject runtime hooks into the office agent interaction loop and propagate Team Memory metrics. |

## Core Contributions

Team Memory is a procedural-memory-oriented layer for existing multi-agent
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

Team Memory is not an individual-agent long-term memory method. The durable
experience is stored in shared procedural memory, while each host MAS keeps its
native execution state:

| MAS | Native agent memory setting in experiments | Paper note |
|---|---|---|
| AutoGen | Short-term conversation/context memory only. | No durable per-agent memory beyond the current run. |
| DyLAN | Short-term conversation/context memory only. | Dynamic debate state is local to the run. |
| AgentNet | AgentNet's built-in agent memory remains enabled. | Treat AgentNet memory as part of the host MAS, not as Team Memory. |

This distinction must be stated in the paper. Comparisons are therefore between
host-MAS memory settings plus optional external memory layers, not between
identical stateless agents in every MAS.

## Evidence Tracks

| Track | Purpose | Paper role |
|---|---|---|
| Cross-Benchmark generality | Test whether Team Memory improves modern long-horizon agent tasks when added to existing MAS frameworks. | Main evidence. |
| SOP-model sensitivity | Test whether results depend on one SOP extraction/validation model. | Robustness evidence. |
| Minimal component ablation | Test whether SOP evolution and divergence alignment each contribute. | Mechanism evidence. |

`TeamMemoryBench` and other deterministic checks are engineering diagnostics.
They may remain in unit tests, but they are not paper experiments.

## Claim-Evidence Map

| Claim | Experiment | Required comparison | Main metrics |
|---|---|---|---|
| Team Memory improves task performance without changing MAS topology. | E1 Cross-Benchmark generality | Same benchmark, MAS, actor, deterministic run configuration, and case; compare No-memory, selected memory baselines, and Team Memory. | Official benchmark score and macro average. |
| SOP extraction results do not depend on a single SOP model. | E2 SOP-model sensitivity | Fixed AutoGen, actor, cases, deterministic run configuration, prompts, and budget; vary only SOP model. | Task score, JSON validity, candidate pass rate, evidence correctness, safety rejection, cost. |
| SOP evolution and divergence alignment are both useful additions. | E3 Minimal ablation | Four matched conditions only under AutoGen. | Task score, SOP reuse, divergence recovery, unsafe accepted, token overhead, latency. |

## E1 Cross-Benchmark Generality

Purpose: test whether Team Memory works as a plug-in procedural-memory layer
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
- SOP model for Team Memory;
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
| ALFWorld | AutoGen, AgentNet, DyLAN | No-memory, selected baselines, Team Memory | Procedural-memory anchor. |
| WebArena | AutoGen, AgentNet, DyLAN | No-memory, selected baselines, Team Memory | Web state, authorization, and tool verification. |
| MultiAgentBench/MARBLE | AutoGen, AgentNet, DyLAN | No-memory, selected baselines, Team Memory | Multi-agent coordination and divergence. |
| OfficeBench | AutoGen, AgentNet, DyLAN | No-memory, selected baselines, Team Memory | Long office workflows and artifact state. |

Current implementation status is tracked separately from the target matrix.
Cells that do not yet inject the host MAS and memory method into the official
single-case runtime must remain blocked, not reported.

The full E1 expansion request is tracked in
`benchmark-results/unified-v3/tables/e1_full_completion_gap_audit.md`, with the
frozen full MultiAgentBench manifest at
`manifests/e1_multiagentbench_full_manifest.json`. That manifest contains all
400 official MultiAgentBench records: 100 Research, 100 Database, 100 Coding,
and 100 Minecraft. Under the requested seed-0 AutoGen/DyLAN x no-memory,
G-Memory, and Team Memory/GEMS matrix, the expanded E1 target has 2760 cells:
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
`team-memory`; WebArena AutoGen with `no-memory` and `team-memory`; and
OfficeBench AutoGen with `no-memory` and `team-memory`. MultiAgentBench/MARBLE
is excluded from the E1 main plan until a Research-only paired plan is
explicitly adopted and smoke-tested. AgentNet is excluded because matched
baselines are missing, and `agent-native-memory`, `generative-memory`, and
`mem0` are excluded because they do not yet have real official-loop injection.
Smoke anchors may use known official cases to validate wiring, but they are not
paper E1 results unless they are part of the final matched subset. ALFWorld E1
uses the official GMemory adapter path for `no-memory`, `gmemory`, and
`team-memory`; that adapter imports Chroma-backed memory backends at module load
time, so the runnable environment includes `langchain==0.3.25`,
`langchain-chroma==0.2.3`, `finch_clust==0.2.0`, and `finchpy==0.0.1`. The
GMemory delegate also applies the same ALFWorld/TextWorld `EvalSymbol`
compatibility patch used by the Team Memory ALFWorld loop before constructing
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
for Team Memory subprocess startup. WebArena observation screenshots use
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
  from Team Memory sidecar metrics.
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
- memory method: Team Memory;
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
   sha256("team-memory-e2-v1" + benchmark_id + official_case_id)
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
| `no-extra-components` | Off | Off | Off | Same MAS and actor model with no added Team Memory mechanism. |
| `blackboard-only` | Off | On | Off | Isolate the structured shared execution record substrate. |
| `sop-only` | On | On | Off | Test procedural memory plus the required blackboard substrate. |
| `divergence-only` | Off | On | On | Test state alignment plus the required blackboard substrate. |
| `full` | On | On | On | Test the complete GEMS configuration. |

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

E3 case IDs use the same deterministic manifest policy as E2, except the hash
salt is:

```text
team-memory-e3-v1
```

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

## Result Tables

The tables below contain the accepted results currently available for this protocol; unevaluated cells remain `TBD`. Legacy ALFWorld results
from the old G-Memory-style protocol are useful for debugging resume/import
logic, but they are not comparable to the new E1/E2/E3 table cells.

The manuscript presentation is integrated through
`paper/main.tex`, with the three editable LaTeX tables directly in the
Experiments section and the 12-model E2 figure in
`paper/image/Fig2-E2-procedural-memory-model-sensitivity.pdf` (also SVG and PNG).
Plotting code is `paper/code/draw-Fig2-E2-procedural-memory-model-sensitivity.py`; the additional effectiveness summary is generated by `paper/code/draw-Fig3-effectiveness-evidence.py`.
`docs/EXPERIMENTS_PAPER_NOTES.md` explains the contribution mapping and metric
definitions. The raw summary views below retain their upstream scores; the
paper omits E1 averages over unequal coverage and E2 mixed-scale Overall.

### T1 Cross-Benchmark Generality

| MAS | Memory method | ALFWorld | WebArena | MultiAgentBench | OfficeBench | Avg. |
|---|---|---:|---:|---:|---:|---:|
| AutoGen | No-memory | 0.500 | 0.125 | TBD | 0.417 | 0.347 |
| AutoGen | Agent-native memory | TBD | TBD | TBD | TBD | TBD |
| AutoGen | Generative Memory | TBD | TBD | TBD | TBD | TBD |
| AutoGen | G-Memory | 0.833 | TBD | TBD | TBD | 0.833 |
| AutoGen | mem0 | TBD | TBD | TBD | TBD | TBD |
| AutoGen | Team Memory (Ours) | 0.500 | 0.167 | TBD | 0.417 | 0.361 |
| AgentNet | No-memory | TBD | TBD | TBD | TBD | TBD |
| AgentNet | Agent-native memory | TBD | TBD | TBD | TBD | TBD |
| AgentNet | Generative Memory | TBD | TBD | TBD | TBD | TBD |
| AgentNet | G-Memory | TBD | TBD | TBD | TBD | TBD |
| AgentNet | mem0 | TBD | TBD | TBD | TBD | TBD |
| AgentNet | Team Memory (Ours) | TBD | TBD | TBD | TBD | TBD |
| DyLAN | No-memory | 0.333 | TBD | TBD | TBD | 0.333 |
| DyLAN | Agent-native memory | TBD | TBD | TBD | TBD | TBD |
| DyLAN | Generative Memory | TBD | TBD | TBD | TBD | TBD |
| DyLAN | G-Memory | 0.417 | TBD | TBD | TBD | 0.417 |
| DyLAN | mem0 | TBD | TBD | TBD | TBD | TBD |
| DyLAN | Team Memory (Ours) | 0.583 | TBD | TBD | TBD | 0.583 |

Values are official single-case scores averaged over the current paper-runnable
matched subset in `benchmark-results/unified-v3/tables/e1_cross_benchmark_generality_summary.md`.
The `Avg.` column averages only reported benchmark cells in the same row;
blocked or excluded cells remain `TBD`.
This row-specific average is not comparable across methods with different
benchmark coverage and is omitted from the main-text table.

### T2 SOP-Model Sensitivity

The fixed setting is AutoGen, actor model `gpt-5-mini`, memory method
`team-memory`, full configuration, and `seed=0` as a compatibility field only.
The E2 master table is stored at
`benchmark-results/unified-v3/tables/e2_sop_model_sensitivity_master.md`, with
machine-readable quality metadata in
`benchmark-results/unified-v3/tables/e2_sop_model_sensitivity_master.json`.
Values below are computed only from accepted official result JSON rows. Smoke
and preflight checks are not paper evidence.

Complete 72-cell SOP models:

| SOP model | ALFWorld | WebArena | MultiAgentBench Research | OfficeBench | Overall | Runs / cases |
|---|---:|---:|---:|---:|---:|---|
| qwen3.5-0.8b | 0.500 | 0.167 | 4.306 | 0.458 | 1.009 | 72 / 72 |
| qwen3.5-2b | 0.500 | 0.167 | 4.250 | 0.458 | 1.000 | 72 / 72 |
| qwen3.5-9b | 0.417 | 0.167 | 4.278 | 0.458 | 0.991 | 72 / 72 |
| qwen3.5-27b | 0.500 | 0.125 | 4.250 | 0.417 | 0.972 | 72 / 72 |
| gemma-4-12b-it | 0.500 | 0.167 | 4.306 | 0.417 | 0.995 | 72 / 72 |
| gemma-4-31b-it | 0.500 | 0.167 | 4.167 | 0.500 | 1.000 | 72 / 72 |
| deepseek-v4-flash-0731 | 0.583 | 0.250 | 4.278 | 0.375 | 1.019 | 72 / 72 |
| gpt-5.6-terra | 0.417 | 0.208 | 4.278 | 0.375 | 0.977 | 72 / 72 |
| qwen3.8-max | 0.500 | 0.208 | 4.361 | 0.417 | 1.019 | 72 / 72 |
| glm-5.2 | 0.500 | 0.208 | 4.306 | 0.458 | 1.023 | 72 / 72 |
| kimi-k3 | 0.500 | 0.208 | 4.250 | 0.375 | 0.986 | 72 / 72 |
| claude-opus-5 | 0.417 | 0.250 | 4.250 | 0.458 | 1.014 | 72 / 72 |

Partial / incomplete SOP models:

None.

`Overall` above is an upstream mixed-scale summary, not a paper aggregate:
Research ratings and the other benchmarks' 0--1 scores have different units.
The main-text table and figure report each benchmark separately. Across the
12 model means, the score ranges are 16.7 pp (ALFWorld), 12.5 pp (WebArena),
0.194 rating points (Research), and 12.5 pp (OfficeBench). These are descriptive ranges across configured model blocks, not repeated-run
confidence intervals. A subsequent audit of all 864 accepted raw records found
zero SOP retrieval/reuse counters throughout and no SOP-model-call or parse-success
counters. The current cross-benchmark runtime does not directly call an extractor.
These results do not identify an executed extraction-model intervention; the
paper therefore does not interpret them as demonstrated SOP-model robustness.

`kimi-k3` and `claude-opus-5` are now complete 72-cell models.
`claude-opus-5` required fenced JSON parsing compatibility in the SOP/preflight
path and a WebArena auto-login navigation timeout compatibility patch; these do
not change official evaluator scoring. Smoke and preflight checks are not paper
evidence.

### T3 Minimal Component Ablation

| Benchmark | Condition | Procedural SOP | Divergence alignment | Official task score | Delta vs no-extra-components | SOP retrievals / case | Divergence events / case | Unsafe accepted | Token overhead | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MultiAgentBench Research | no-extra-components | Off | Off | 4.194 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |
| MultiAgentBench Research | sop-only | On | Off | 4.333 | 0.139 | 0.000 | 0.000 | 0.000 | 427.167 | n/a |
| MultiAgentBench Research | divergence-only | Off | On | 4.222 | 0.028 | 0.000 | 1.667 | 0.000 | -550.500 | n/a |
| MultiAgentBench Research | full | On | On | 4.444 | 0.250 | 0.000 | 1.667 | 0.000 | 28762.667 | n/a |
| OfficeBench | no-extra-components | Off | Off | 0.389 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | 0.000 |
| OfficeBench | sop-only | On | Off | 0.333 | -0.056 | 0.000 | 0.000 | 0.000 | n/a | 0.000 |
| OfficeBench | divergence-only | Off | On | 0.333 | -0.056 | 0.000 | 26.556 | 0.000 | n/a | 0.000 |
| OfficeBench | full | On | On | 0.333 | -0.056 | 0.000 | 28.111 | 0.000 | n/a | 0.000 |
| WebArena | no-extra-components | Off | Off | 0.056 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | 212.024 |
| WebArena | sop-only | On | Off | 0.000 | -0.056 | 0.000 | 0.000 | 0.000 | n/a | 147.701 |
| WebArena | divergence-only | Off | On | 0.000 | -0.056 | 0.000 | 10.444 | 0.000 | n/a | 189.219 |
| WebArena | full | On | On | 0.056 | 0.000 | 0.000 | 9.444 | 0.000 | n/a | 189.310 |
| ALFWorld | no-extra-components | Off | Off | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| ALFWorld | sop-only | On | Off | 0.417 | 0.083 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| ALFWorld | divergence-only | Off | On | 0.500 | 0.167 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| ALFWorld | full | On | On | 0.500 | 0.167 | 0.000 | 0.000 | 0.000 | n/a | n/a |

The E3 master table is stored at
`benchmark-results/unified-v3/tables/e3_component_ablation_master.md`. Scores
are averaged only over matched case IDs within each benchmark. Failed runs,
sidecars, partial rows, and diagnostic-only MARBLE fields are excluded.

ALFWorld and MultiAgentBench Research show positive full/no-extra gains with no
regression in their matched case comparisons. WebArena has the same mean score,
while OfficeBench decreases from 0.389 to 0.333. The source master's
`metric_notes` identifies the former `SOP reuse success` column as mean
`sop_retrieval_count`, and the former `Divergence recovery` column as mean
`divergence_events`; the corrected headings above reflect those definitions.
Zero retrieval counts do not demonstrate successful reuse. `unsafe_accepted`
is zero across all E3 blocks, but without counts of unsafe update opportunities
it does not establish safety rejection effectiveness. Divergence-event counts
are activity diagnostics, not recovery accuracy.

#### Main-text contribution view

| Condition | SOP (C1) | Blackboard (C2) | Alignment (C3) | ALFWorld (%) | WebArena (%) | Research rating | OfficeBench (%) |
|---|---|---|---|---:|---:|---:|---:|
| No added components | Off | Off | Off | 33.3 | 5.6 | 4.194 | 38.9 |
| SOP only | On | On | Off | 41.7 | 0.0 | 4.333 | 33.3 |
| Alignment only | Off | On | On | 50.0 | 0.0 | 4.222 | 33.3 |
| Full | On | On | On | 50.0 | 5.6 | 4.444 | 33.3 |

| Contrast from the completed four-condition diagnostic table | ALFWorld (pp) | WebArena (pp) | Research rating points | OfficeBench (pp) |
|---|---:|---:|---:|---:|
| Full − no added components: joint system | +16.7 | 0.0 | +0.250 | −5.6 |
| Full − alignment only: C1 conditional on C2+C3 | 0.0 | +5.6 | +0.222 | 0.0 |
| Full − SOP only: C3 conditional on C1+C2 | +8.3 | +5.6 | +0.111 | 0.0 |

The completed table above is the prior four-condition diagnostic result. It
does not isolate the blackboard as an independent treatment, and zero recorded
SOP retrievals prevent assigning differences specifically to successful reuse of
evolved procedures. The next E3 protocol adds `blackboard-only`; those rows must
be run with matched case IDs before any five-condition ablation is reported as
paper evidence.

WebArena excludes the official runtime/no-score replacement chain `732`, `722`,
`646`, `27`, `410`, `30`, `601`, and `409`; partial rows from excluded
replacements `722` and `601` are not counted. OfficeBench excludes official
evaluator bug cases `3-15/0` and `3-65/0`, with replacements `3-66/0` and
`3-60/0`. MultiAgentBench uses the Research-focused subset because Research has
stable official `task_evaluation` scoring while Database and Coding remain
blocked at the environment/scoring layer.

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

## Final Status and Evidence Boundary

The final experiment audit is stored at
`benchmark-results/unified-v3/tables/final_experiment_audit.md`, with
machine-readable metadata at
`benchmark-results/unified-v3/tables/final_experiment_audit.json`.

Current audited status:

- E1 Cross-Benchmark Generality: 168 / 168 accepted cells.
- E2 SOP-Model Sensitivity: 12 complete 72-cell SOP-model blocks.
- E3 Minimal Component Ablation: previous four-condition diagnostic table has
  240 / 240 case-condition cells, summarized as 16 benchmark-condition rows.
  The next five-condition protocol requires new `blackboard-only` matched rows
  before it is complete.

Evidence boundary:

- Smoke and preflight checks are engineering checks, not paper evidence.
- Runtime/no-score runs, failed runs, sidecars, and partial rows are not counted
  as accepted paper results.
- `kimi-k3` and `claude-opus-5` are complete 72-cell E2 models.
- The `claude-opus-5` run uses SOP/preflight fenced JSON parsing compatibility
  and WebArena auto-login timeout compatibility only; no official evaluator
  scoring logic is changed.
- WebArena AWS state must be controlled by the run plan; audits must not trigger
  WebArena cells or treat deferred WebArena cells as scored failures.

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
