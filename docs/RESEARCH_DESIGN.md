# Research Design

This document maps the implementation to the paper method. It is a scientific
contract, not a command manual.

## Memory Layers

The framework maintains three separated layers:

```text
M_priv: each agent's private role facts and current local task.
M_task: lightweight shared blackboard, evidence, canonical goal, and task graph.
M_proc: versioned procedural memory.
```

The shared blackboard stores public execution records:

```text
e_t = {task, observation, action, result/outcome, error}
```

Implementations may attach evidence, state keys, goals, and graph snapshots as
structured metadata. They must not store hidden chain-of-thought as shared state.

A procedural memory item is:

```text
S_i = (P_i, M_i, G_i)
```

where `P_i` is the procedure, `M_i` is metadata, and `G_i` is the procedure
dependency graph. Version numbers are immutable storage metadata.

## SOP Evolution

SOP candidates support create, update, delete, and context-specific variants.
Candidates are first valued by:

```text
R(S) = Success(S) - lambda * Cost(S)
```

Utility alone is insufficient. The safety validator rejects any update whose
risk exceeds threshold or whose procedure removes mandatory confirmation,
authorization, backup, or irreversible-action protections:

```text
Risk(S_i -> S_i') < tau
```

An update is written only if all gates pass:

```text
Promote(c) =
  Utility(c) and Safe(c) and Verified(c) and Causal(c)
  and Validated(c) and Benefit(c)
```

Gate meanings:

| Gate | Meaning |
|---|---|
| Utility | Declared `Success - lambda * Cost` is non-negative. |
| Safe | Risk is below threshold and mandatory safety steps are preserved. |
| Verified | Key states are backed by verifiable evidence. |
| Causal | The candidate procedure, not unrelated luck or another agent, explains the improvement. |
| Validated | A qualified validation trial succeeds. |
| Benefit | `candidate_reward - baseline_reward - cost > 0`. |

Narrow optimizations are stored as `variant_of` plus `context_conditions`, not
as destructive overwrites of a general SOP. Published SOP updates use
`base_version` compare-and-swap, so stale concurrent writes are rejected.

## Shared Blackboard

The blackboard is a lightweight bridge from short-term collaboration state to
long-term procedural memory.

```text
Execution error -> Blackboard error record -> SOP warning / constraint
Task dependency graph -> Validated procedure graph -> SOP graph
```

This design keeps communication compact while preserving enough public evidence
for procedure extraction, state verification, and later audit.

## Divergence Alignment

The Manager-Agent or native framework coordinator detects and resolves three
task-understanding divergences.

Goal divergence combines semantic distance and a consistency judge:

```text
D_g = lambda * (1 - cos(g_i, g*)) + (1 - lambda) * J_goal(g_i, g*)
```

Plan divergence compares task dependency graphs:

```text
D_p = GED(G_i, G*) / max(|G_i|, |G*|)
```

State divergence detects conflicting claims over the same state key. State
resolution follows:

```text
Authoritative State > Tool Observation > Verified Artifact > Agent Inference
```

The alignment workflow is:

```text
Detect -> Classify -> Verify -> Resolve -> Realign
```

Agent negotiation is used only when no verifiable external truth source exists.

## Retrieval

Future tasks recall SOPs through:

```text
Candidate(q) = BM25(q) union VectorSearch(q)
```

BM25 covers SOP title, description, applicability, context conditions, tags, and
step text. Vector search covers the same SOP document semantically. Procedure
graphs remain part of SOP representation and divergence analysis, but the reuse
path is intentionally simple and auditable.

## Implementation Boundaries

- `HashingEmbedder` is a deterministic offline baseline, not a publication
  embedding claim.
- Plan graph distance is a fast graph-edit proxy unless an exact GED solver is
  substituted.
- Causal fields must come from experiment design or external evidence, not SOP
  self-assessment alone.
- SQLite is suitable for single-node research runs; distributed runs should use
  a shared transactional database.
