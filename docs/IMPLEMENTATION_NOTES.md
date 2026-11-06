# Implementation Notes

This file documents how the benchmark harnesses in this repository implement the protocol of
the paper, including the points where a harness necessarily departs from it.

## Plug-in architecture

Every memory is a plug-in behind one contract ([`src/sepm/plugins/base.py`](../src/sepm/plugins/base.py)):
`begin_task → context → record* → notices → end_task`, plus `abandon_task` when a host never
observes an outcome. Hosts call only these hooks, so the host is identical across memory conditions.

| Plug-in | Memory |
|---|---|
| `SEPMPlugin` | SEPM (`ProceduralEpisode`) with an LLM goal judge (J<sub>goal</sub>) and an LLM procedure curator |
| `GMemoryFamilyPlugin` | the official G-Memory, Generative Agents, and MetaGPT memory classes from `external/GMemory/mas/memory`, unchanged |
| `Mem0Plugin` | the official `mem0ai` library with automatic memory writing |
| `NoMemory` | no-op |

## Shared implementation choices

1. **Completion budget.** `gpt-5-mini` counts hidden reasoning tokens against
   `max_completion_tokens`. With the protocol caps alone (4096 for ImplicitMemBench, 1024/4096 for
   MultiAgentBench), some replies are empty, so every call receives 4× the protocol budget.
2. **Empty replies** are retried once, as in ImplicitMemBench's own client.
3. **Transient endpoint errors** are retried. Judge errors make an item fail so that it is re-run,
   instead of being scored as wrong.
4. **Embeddings.** Without a local GPU, all memories use `text-embedding-3-small` through the API.
   On Linux, the G-Memory-family baselines keep G-Memory's default `all-MiniLM-L6-v2`.
5. **Vector store.** On Windows, `LiteChroma` replaces Chroma with the same API subset and the same
   squared-L2 distance, so G-Memory's `similarity = 1 - distance` thresholds behave as upstream.
6. **Third-party syntax fixes (behavior-preserving).** `scripts/patch_gmemory_py311.py` rewrites four
   PEP 701 f-strings in G-Memory so it imports on Python 3.11; `scripts/patch_marble.py` fixes a
   syntax error in MARBLE's `Evaluator.parse_research_ratings`.

## SEPM on free-text hosts

* Free-text actions are matched to procedure steps by embedding cosine similarity (threshold 0.5,
  fixed in advance). A message that matches no step counts as conversation, not as an off-plan
  action; plan divergence is raised for skipped prerequisites of matched steps.
* As in Sec. 4.3 and Supp. B.5, failed steps of an episode become conditional warnings; a warning
  must cite a failed step.

## ImplicitMemBench

* Host: an AutoGen-style group chat (manager → assistant → critic, at most one revision;
  [`src/sepm/hosts/autogen_team.py`](../src/sepm/hosts/autogen_team.py)). Only the assistant's
  final message is user-facing and scored.
* The learning or priming turns and the interference turns of an item form one exposure episode
  recorded into the memory (step = scripted assistant turn, observation = the feedback or user
  turn that follows). The team then answers the probe with the memory's context.
* Failure signals for SEPM's step outcomes come from a fixed marker list read off the item format
  (`domains.IMPLICIT_MEMORY_FAILURE_MARKERS`), not tuned on results.
* The judge, rule-based checks, and priming judge are imported from the benchmark's
  `src/evaluation/evaluation.py`; judge = `gpt-4o-mini`; temperature 0 (procedural memory and
  conditioning) and 0.8 (priming probe).
* Interference lengths: ~200 tokens truncates the original interference turns; ~1000 and ~2000
  append neutral `gpt-4o-mini` turns that reuse no learning-phase names or symbols (cached per item).
* Accumulate setting: one SEPM pool across all items, run sequentially in the original or a
  shuffled (seed 0) order; "foreign" counts injected procedures created by another item.

## MultiAgentBench

* Agent calls use `gpt-5-mini`, evaluator calls use `claude-opus-5`, at most 3 iterations,
  temperature 0.7 (0.1 for coding).
* TS: research = mean(innovation, safety, feasibility) × 20; coding = mean of MARBLE's
  `evaluate_code_quality` criteria × 20; database = matched fraction of gold root causes × 100 with
  the official label extraction (`scripts/database/batch_eval.py`).
* CS = (mean communication + mean planning) / 2 × 20, as in MARBLE's `batch_eval.py`. The released
  graph loop has these evaluations disabled and its prompt templates break `str.format`, so the
  harness fills the named placeholders and uses MARBLE's own `parse_score`.
* Tool schemas are repaired for strict endpoints (empty `enum`, `required` keys without a property).
* Memory feedback: a task counts as successful for memory updates when TS ≥ 60 (reward = TS/100),
  for every method.
* The research tool `get_related_papers` uses KeyBERT; a stand-in implements KeyBERT's default
  algorithm with API embeddings, so no PyTorch install is needed.
