# Re-run notes: implementation choices and deviations

This file records how the re-run experiments are implemented and where they necessarily
differ from the protocol stated in the paper. Numbers produced under these notes are
reported as measured; nothing is copied from `paper/`.

## Scope

| Experiment | Where | Status |
|---|---|---|
| E4/E5 ImplicitMemBench (gpt-5-mini) | Windows (local) | running |
| E3 MultiAgentBench coding + research | Windows (local) | running |
| E3 MultiAgentBench database | Linux (`docs/LINUX_RUNBOOK.md`) | pending |
| E1, E6, E8 (LOO, cross-host), E9 (stats, sensitivity), E10, E12, E13 | Linux | pending |
| Table 3 / Fig. 11 / qwen transfer, WebArena, TDW-MAT/C-WAH, Qwen2.5-7B | — | excluded by request (no GPU / deferred) |
| LEGOMem, MATM, AWM†, Memp†, Synapse†, LLMA-Mem rows; E2 entirely | — | not run (no official code; user decision) |
| E9 detection precision / κ (human annotation) | — | not run |
| ImplicitMemBench "Mem0 + Key Info" oracle | — | not run: the released items contain no key-information field, so the oracle input is undefined |

## Plug-in architecture

Every memory is a plug-in behind one contract (`src/sepm/plugins/base.py`):
`begin_task → context → record* → notices → end_task` (+ `abandon_task` when a host never
observes an outcome). Implementations:

* `SEPMPlugin` – SEPM (`ProceduralEpisode`) with an LLM goal judge (J_goal, gpt-5-mini),
  an LLM SOP curator (gpt-5-mini) and API embeddings.
* `GMemoryFamilyPlugin` – the **official** G-Memory, Generative Agents and MetaGPT memory
  classes from `external/GMemory/mas/memory`, unchanged.
* `Mem0Plugin` – the official `mem0ai` library (automatic memory writing).
* `NoMemory` – no-op.

Hosts call only these hooks, so the host is identical across memory conditions.

## Shared implementation choices

1. **Model budget.** gpt-5-mini counts hidden reasoning tokens against `max_completion_tokens`.
   With the protocol caps alone (4096 for ImplicitMemBench, 1024/4096 for MultiAgentBench),
   15% of single-LLM ImplicitMemBench replies were empty and MARBLE coding produced empty
   `solution.py` files. All calls therefore get 4× the protocol budget. Runs made before this
   change are archived in `benchmark-results/implicitmem/discarded_budget4096/` and not reported.
2. **Empty replies** are retried once (ImplicitMemBench's own client does the same).
3. **Transient relay errors** (connection drops, a relay-side `unsupported_parameter` 400 on
   parameters it rewrote itself) are retried; judge errors make the item fail and it is re-run,
   instead of being scored as wrong (the benchmark's judge helpers return "Incorrect" on error).
4. **Embeddings.** Locally all memories use `text-embedding-3-small` through the API (no torch
   on the 4 GB workstation). On Linux, G-Memory-family baselines keep G-Memory's default
   `all-MiniLM-L6-v2` (`EmbeddingFunc`) inside the official adapter.
5. **Vector store.** Locally, `LiteChroma` replaces Chroma with the same API subset and the same
   default squared-L2 distance (so G-Memory's `similarity = 1 - distance` thresholds behave as
   upstream). Linux uses real Chroma.
6. **Third-party syntax fixes (behaviour-preserving).** `scripts/patch_gmemory_py311.py` rewrites
   four PEP 701 f-strings in G-Memory so it imports on Python 3.11; `scripts/patch_marble.py`
   fixes a syntax error in MARBLE's `Evaluator.parse_research_ratings`.

## SEPM on free-text hosts

* Free-text actions are matched to procedure steps by embedding cosine similarity (threshold 0.5,
  fixed in advance). A message that matches no step counts as conversation, not an off-plan
  action; plan divergence is raised for skipped prerequisites of matched steps.
* Following §4.4 / App. B.5 ("failures can yield conditional warnings … attached as soon as their
  supporting record exists"), candidate extraction also turns failed steps of the episode into
  conditional warnings; a warning must cite a failed step.

## ImplicitMemBench (E4/E5)

* Host: an AutoGen-style group chat (manager → assistant → critic, ≤1 revision; `src/sepm/hosts/autogen_team.py`).
  Only the assistant's final message is user-facing and scored.
* Each item's learning/priming + interference turns form one exposure episode recorded into the
  memory (step = scripted assistant turn, observation = the feedback/user turn that follows).
  The exposure counts as successful unless its last feedback is a failure signal. Then the team
  answers the probe with the memory's context; the probe episode is abandoned (no outcome).
* Failure signals for SEPM's step outcomes: a fixed marker list read off the item format
  (`domains.IMPLICIT_MEMORY_FAILURE_MARKERS`), not tuned on results.
* Judge, rule-based checks and priming judge are imported from the benchmark's
  `src/evaluation/evaluation.py`; judge = gpt-4o-mini; T = 0 (Proc/Cond), 0.8 (priming probe).
* Interference lengths: ~200 truncates the original interference turns; ~1000/~2000 append neutral
  gpt-4o-mini turns that must not reuse learning-phase names or symbols (cached per item).
* Accumulate setting: one SEPM pool across all items, items run sequentially in original or
  shuffled (seed 0) order; "foreign" counts injected procedures created by another item.

## MultiAgentBench (E3)

* Host: **MARBLE's own engine** (graph coordination, official agent profiles, tools and
  communication). The paper states AutoGen; MARBLE has no AutoGen mode, so memories plug into
  each agent's `BaseMemory` slot, which MARBLE inserts into act/plan prompts and updates after
  every action.
* All agent calls → gpt-5-mini (agents may name other models in tool arguments; ignored), all
  evaluator calls → claude-opus-5, max 3 iterations, T = 0.7 (coding 0.1).
* TS: research = mean(innovation, safety, feasibility) × 20; coding = mean of MARBLE's
  `evaluate_code_quality` criteria × 20 (an empty/missing `solution.py` scores the evaluator's
  failure default, 1s); database = matched fraction of gold root causes × 100 with the official
  gpt-4o-mini label extraction (`scripts/database/batch_eval.py`).
* CS = (mean communication + mean planning) / 2 × 20, as in MARBLE's `batch_eval.py`. The
  released graph loop has these evaluations disabled and its prompt templates break
  `str.format` (literal JSON braces), so the harness fills the named placeholders and uses
  MARBLE's own `parse_score`. Iterations without communication contribute no communication score.
* Tool schemas are repaired for strict endpoints (empty `enum`, `required` keys without a property).
* Memory feedback: a task counts as successful for memory updates when TS ≥ 60 (reward = TS/100),
  for every method.
* The research tool `get_related_papers` uses KeyBERT; a stand-in implements KeyBERT's default
  algorithm with API embeddings (no torch).
* Truncated tool-call arguments are counted per case (`truncated_tool_calls`).
