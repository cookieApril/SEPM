<h1 align="center">SEPM: Self-Evolving Procedural Memory Grounded in Task Graphs for Multi-Agent Collaboration</h1>

<!-- Replace the arXiv homepage with the SEPM abstract URL when available. -->
<div align="center">
  <a href="https://arxiv.org/">
    <img src="https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv" alt="arXiv">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="Apache 2.0 license">
  </a>
</div>


---

## 📑 Contents

- [👀 Overview](#-overview)
- [✨ Main Results](#-main-results)
- [🔎 Case Study: Complete Workflow](#-case-study-complete-workflow)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [🧪 Evaluation](#-evaluation)
- [📄 Citation and License](#-citation-and-license)

---

## 👀 Overview

**In multi-agent work, memory and collaboration cannot be separated.** A remembered step is useful only when the team knows who should perform it, which other agents' outputs it depends on, and what evidence confirms that its prerequisites are satisfied. Sharing a procedure or assigning it to a role does not establish these conditions. When they are left implicit, agents may repeat work, act before a handoff is ready, or reuse a procedure after its assumptions have changed.

We propose **SEPM**, a complete memory-based collaboration framework that makes **task dependency graphs** the common representation for persistent memory and live execution. Nodes describe executable steps; directed edges record cross-agent prerequisites. A retrieved procedure becomes a task graph linked to agent assignments and verified state on a shared blackboard. The same edges then support prerequisite checks, plan comparison, and the propagation of corrections when an agent's goal, plan, or world-state view diverges.

The framework separates private agent context, a shared execution blackboard, and a persistent procedure pool, while the task graph connects them across the task lifecycle:

- **Store and evolve:** retain verified actions, dependency edges, applicability conditions, and conditional warnings from failures in versioned procedures; supported repairs refine a procedure or create a new one.
- **Retrieve and use:** bring relevant procedures into planning so a new team reuses both the steps and the coordination constraints that made them work.
- **Coordinate and verify:** link agent assignments and blackboard evidence to graph nodes, detect goal/plan/state divergence, and correct affected steps before their dependents proceed.

The memory that guides collaboration is therefore revised by evidence from that collaboration.

<p align="center">
  <a href="docs/readme-assets/framework.png"><img src="docs/readme-assets/framework.png" width="95%" alt="SEPM architecture with task graph, blackboard, divergence detection, and evolving procedures"></a><br>
  <sub><strong>SEPM framework.</strong> Retrieved procedures guide the task graph; blackboard evidence supports divergence-aware alignment and feeds verified revisions back into procedural memory.</sub>
</p>

The framework diagram follows one full cycle: retrieve a reusable procedure, instantiate its dependencies for the current team, record execution against those dependencies, correct deviations, and update the procedure pool using verified outcomes. It shows why the graph serves both collaboration and memory evolution.


<p>
  <a href="docs/readme-assets/multiagentbench.png"><img align="left" hspace="1" vspace="6" src="docs/readme-assets/multiagentbench.png" width="47%" alt="Task and communication scores on MultiAgentBench coding, research, and database tasks"></a>
  MultiAgentBench measures both the quality of a team's output and the quality of its communication across coding, research, and database work.
  <br><br>
  <strong>Task quality and communication.</strong> Under AutoGen, SEPM has the highest Task Score (TS) and Communication Score (CS) among the compared methods in all three environments. TS measures output quality; CS measures communication and planning. The chart groups the three environments within each metric so the two outcomes can be compared at a glance.
  <br><br>
  Each environment contains <strong>100 tasks</strong>. Relative to G-Memory, SEPM's largest TS gain is on database tasks (<strong>+7.30 points</strong>), while its largest CS gain is on research tasks (<strong>+11.14 points</strong>). Both metrics retain the benchmark's reporting scale.
</p>
<br clear="all">

The gains across both measures illustrate the aim of the framework: memory should help agents produce better results while coordinating the steps that lead to those results.

### Key Contributions

1. **Coordination-native procedural memory.** Task dependency graphs represent both active collaborative plans and reusable procedures, so cross-agent prerequisites survive the transition from one task to the next.
2. **Evidence-grounded alignment.** A shared blackboard records structured execution evidence; goal, plan, and world-state divergence signals guide corrections while agents retain their private context.
3. **Self-evolving procedures.** Verified actions and dependencies can become graph-structured SOPs. Revisions, context-specific variants, and retirement are governed by validation, utility, safety, and version history.

The repository implements these ideas as an installable Python memory layer:

| Mechanism | Role in the workflow | Code |
| --- | --- | --- |
| Shared task graph and blackboard | Store assignments, prerequisites, observations, outcomes, errors, and evidence | [`models.py`](src/sepm/models.py), [`storage.py`](src/sepm/storage.py) |
| Divergence and realignment | Detect goal, plan, and state disagreements; resolve state claims by evidence priority | [`divergence.py`](src/sepm/divergence.py), [`evidence.py`](src/sepm/evidence.py) |
| Evolving procedures | Propose, validate, reproduce, and version graph-structured SOPs | [`service.py`](src/sepm/service.py), [`safety.py`](src/sepm/safety.py) |
| Contextual retrieval | Filter applicable SOPs and rank them with lexical and vector relevance | [`retrieval.py`](src/sepm/retrieval.py) |

This implementation map follows a procedure from storage and retrieval into execution, alignment, and revision. [`SEPMService`](src/sepm/service.py) connects the modules, while the [storage schema](docs/STORAGE_SCHEMA.md) documents the graph objects, blackboard events, and SQLite version history.

---

## ✨ Main Results

The following comparisons examine SEPM across host systems, model configurations, component choices, and token cost. [Evaluation](#-evaluation) explains the scope of the code and local experiment matrix in this workspace.

### Cross-benchmark task performance

SEPM achieves the highest task score among the compared methods in all six benchmark–host settings. Under AutoGen, the gains over the same host without added memory are **+29.85 points** on ALFWorld, **+25.86** on WebArena, and **+13.00** on OfficeBench.

<table width="100%">
  <thead>
    <tr><th>Host</th><th>Memory</th><th>ALFWorld</th><th>WebArena</th><th>OfficeBench</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="5" align="center" valign="middle"><strong>AutoGen</strong></td><td>No added memory</td><td align="right">59.70</td><td align="right">12.44</td><td align="right">41.67</td></tr>
    <tr><td>Generative Agents</td><td align="right">85.82</td><td align="right">15.27</td><td align="right">38.33</td></tr>
    <tr><td>MetaGPT</td><td align="right">78.36</td><td align="right">25.99</td><td align="right">45.33</td></tr>
    <tr><td>G-Memory</td><td align="right">83.58</td><td align="right">30.30</td><td align="right">50.00</td></tr>
    <tr><td><strong>SEPM</strong></td><td align="right"><strong>89.55</strong></td><td align="right"><strong>38.30</strong></td><td align="right"><strong>54.67</strong></td></tr>
    <tr><td rowspan="5" align="center" valign="middle"><strong>DyLAN</strong></td><td>No added memory</td><td align="right">53.73</td><td align="right">14.90</td><td align="right">33.33</td></tr>
    <tr><td>Generative Agents</td><td align="right">65.67</td><td align="right">15.27</td><td align="right">33.00</td></tr>
    <tr><td>MetaGPT</td><td align="right">49.25</td><td align="right">22.41</td><td align="right">35.00</td></tr>
    <tr><td>G-Memory</td><td align="right">61.94</td><td align="right">25.49</td><td align="right">42.67</td></tr>
    <tr><td><strong>SEPM</strong></td><td align="right"><strong>68.66</strong></td><td align="right"><strong>29.80</strong></td><td align="right"><strong>47.33</strong></td></tr>
  </tbody>
</table>

<sub><strong>Task performance across AutoGen and DyLAN.</strong> Scores are percentages; bold marks the best result within each host and benchmark.</sub>

SEPM leads the strongest alternative by **3.73 / 8.00 / 4.67** points under AutoGen and **2.99 / 4.31 / 4.66** under DyLAN (ALFWorld / WebArena / OfficeBench).

The unweighted three-benchmark mean rises from **37.94% to 60.84%** under AutoGen and from **33.99% to 48.60%** under DyLAN. A textual-procedure control retains the step descriptions and prerequisites but replaces graph objects and adjacency relations with text. Explicit graphs improve its scores by **17.16**, **8.13**, and **12.34** points on ALFWorld, WebArena, and OfficeBench.

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="docs/readme-assets/overall-performance.png"><img src="docs/readme-assets/overall-performance.png" width="65%" alt="Average task scores for memory methods under AutoGen and DyLAN"></a><br>
      <sub><strong>Cross-benchmark mean task score.</strong> Average performance across ALFWorld, WebArena, and OfficeBench within each host.</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <a href="docs/readme-assets/graph-vs-text.png"><img src="docs/readme-assets/graph-vs-text.png" width="100%" alt="Task score comparison between textual SOPs and explicit dependency graphs"></a><br>
      <sub><strong>Explicit graphs versus textual procedures.</strong> Task scores when dependency edges are represented as graph objects or described in text.</sub>
    </td>
  </tr>
</table>

The left panel makes the improvement across both hosts visible in one view. The right panel tests the representation itself: retaining procedure text and prerequisites without explicit graph operations lowers performance on all three benchmarks. Together, the panels connect the overall gain to SEPM's central design choice.

### Model sensitivity

Across the 13 reported configurations, task scores span **57.46–93.28%** on ALFWorld, **14.53–43.23%** on WebArena, and **45.67–67.67%** on OfficeBench. The task-execution and procedural-memory endpoints change together; the planning agent and goal-consistency judge stay fixed.

<details>
<summary><strong>Show all model-sensitivity scores</strong></summary>

<table width="100%">
  <thead><tr><th>Model endpoint</th><th>ALFWorld</th><th>WebArena</th><th>OfficeBench</th></tr></thead>
  <tbody>
    <tr><td>qwen3.5-0.8b</td><td align="right">58.21</td><td align="right">14.53</td><td align="right">45.67</td></tr>
    <tr><td>qwen3.5-2b</td><td align="right">57.46</td><td align="right">14.78</td><td align="right">45.67</td></tr>
    <tr><td>qwen3.5-9b</td><td align="right">67.16</td><td align="right">16.01</td><td align="right">49.67</td></tr>
    <tr><td>qwen3.5-27b</td><td align="right">81.34</td><td align="right">30.42</td><td align="right">55.67</td></tr>
    <tr><td>gemma-4-12b-it</td><td align="right">72.39</td><td align="right">22.54</td><td align="right">53.00</td></tr>
    <tr><td>gemma-4-31b-it</td><td align="right">77.61</td><td align="right">28.69</td><td align="right">54.33</td></tr>
    <tr><td>deepseek-v4-flash-0731</td><td align="right"><strong>93.28</strong></td><td align="right">38.05</td><td align="right">64.33</td></tr>
    <tr><td>qwen3.8-max</td><td align="right">89.55</td><td align="right">39.41</td><td align="right">63.67</td></tr>
    <tr><td>glm-5.2</td><td align="right">85.82</td><td align="right">37.93</td><td align="right">60.67</td></tr>
    <tr><td>kimi-k3</td><td align="right">90.30</td><td align="right">42.36</td><td align="right">65.33</td></tr>
    <tr><td>gpt-5-mini (main)</td><td align="right">89.55</td><td align="right">38.30</td><td align="right">54.67</td></tr>
    <tr><td>gpt-5.6-terra</td><td align="right">91.79</td><td align="right"><strong>43.23</strong></td><td align="right"><strong>67.67</strong></td></tr>
    <tr><td>claude-opus-5</td><td align="right">92.54</td><td align="right">42.49</td><td align="right">65.33</td></tr>
  </tbody>
</table>

<sub><strong>Joint execution/memory model sensitivity.</strong> Task scores are percentages; bold marks the highest observed value in each benchmark.</sub>

`gpt-5-mini` is the main cross-benchmark setting. `deepseek-v4-flash-0731` leads on ALFWorld, while `gpt-5.6-terra` leads on WebArena and OfficeBench.

</details>

### Component contributions

The shared blackboard is the foundation for both memory and alignment in this ablation. The full configuration leads on all three tasks; the individual gains differ by environment.

<table width="100%">
  <thead><tr><th>AutoGen configuration</th><th>ALFWorld</th><th>WebArena</th><th>OfficeBench</th></tr></thead>
  <tbody>
    <tr><td>No added components</td><td align="right">59.70</td><td align="right">12.44</td><td align="right">41.67</td></tr>
    <tr><td>Blackboard only</td><td align="right">69.40</td><td align="right">27.71</td><td align="right">45.00</td></tr>
    <tr><td>Blackboard + memory</td><td align="right">86.57</td><td align="right">33.99</td><td align="right">45.33</td></tr>
    <tr><td>Blackboard + alignment</td><td align="right">82.09</td><td align="right">34.48</td><td align="right">51.00</td></tr>
    <tr><td><strong>Full SEPM</strong></td><td align="right"><strong>89.55</strong></td><td align="right"><strong>38.30</strong></td><td align="right"><strong>54.67</strong></td></tr>
  </tbody>
</table>

<sub><strong>Component ablation under AutoGen.</strong> Task scores are percentages. Memory extracts, revises, and reuses procedures across tasks; alignment detects and resolves divergence.</sub>

The full system leads all three tasks. The blackboard alone improves on the base configuration, while memory and alignment provide further gains when combined with it.

### Performance and token cost

<p>
  <a href="docs/readme-assets/token-cost.png"><img align="left" hspace="16" vspace="3" src="docs/readme-assets/token-cost.png" width="45%" alt="ALFWorld success versus total token consumption under AutoGen"></a>
  The cost comparison sums input and output tokens over the complete 134-task ALFWorld evaluation under AutoGen. Each point pairs a method's task score with the tokens from that same run.
  <br><br>
  <strong>ALFWorld performance and token cost.</strong> SEPM reaches <strong>89.55%</strong> success with <strong>5.2 million</strong> tokens. The no-memory run uses 4.4M tokens at 59.70% success; Generative Agents uses 5.6M at 85.82%; G-Memory uses 5.7M at 83.58%. The image places these methods in a common success–cost view.
  <br><br>
  The plotted totals cover the entire evaluation, rather than one representative task. They include coordination and memory operations together with task execution.
 </p>
<br clear="all">
  Compared with no added memory, SEPM uses about **18.2%** more tokens and raises success by **29.85 points**. Compared with Generative Agents and G-Memory, it uses about **7.1%** and **8.8%** fewer tokens while improving success by **3.73** and **5.97** points, respectively.

---

## 🔎 Case Study: Complete Workflow

The cleaning-and-storage case traces an end-to-end cycle of **retrieval → collaboration → realignment → memory revision**. The task is to wash dirty cups and dishes in a kitchen and put them in a cabinet.

<p align="center">
  <a href="docs/readme-assets/case-study.png"><img src="docs/readme-assets/case-study.png" width="98%" alt="Complete SEPM cleaning and storage workflow from retrieved procedure to revised memory"></a><br>
  <sub><strong>Cleaning and storage workflow.</strong> The upper path follows planning and execution; the lower path connects detected deviations to an updated procedure.</sub>
</p>

1. **Retrieve a procedure and instantiate the task graph.** The team recalls *Kitchenware Cleaning and Storage v1.0*. Its steps assign cup cleaning, dish cleaning, and cabinet storage, while its graph and warning preserve the prerequisite that items must be cleaned before storage.
2. **Assign work with explicit dependencies.** Agent A finds and cleans cups, Agent B finds and cleans dishes, and Agent C inspects the cabinet and handles final storage. Cabinet inspection can run while A and B clean; handing an item to C depends on verified cleaning by its owner.
3. **Record execution evidence.** A reports finding cups and cleaning them. B cannot find dishes, notices an apple, and starts to put it in the refrigerator. C requests a cup after it has been found. Each agent's subtask, observation, action, outcome, and error are recorded against the shared task state.
4. **Detect two different deviations.** B's refrigerator action is outside the assigned dish-cleaning plan, so SEPM detects **plan divergence**. C's request exposes a **state/prerequisite issue**: finding a cup establishes its location, but does not prove it has been cleaned. The storage handoff cannot proceed from a location claim alone.
5. **Realign without discarding valid progress.** B returns to the dish-cleaning subtask; seeing the apple remains an observation rather than a new task. C may continue inspecting the cabinet, but waits for evidence of A's cleaning outcome before storage. The team resumes from its last verified checkpoint instead of restarting the entire workflow.
6. **Revise and reuse memory.** Verified actions and supported dependency edges inform *v1.1*. The revision retains cleaning-before-handoff and adds conditional guidance: unrelated items do not change the goal, disputed locations require verifiable environmental evidence, “found” is not “cleaned,” and corrected work resumes from a verified checkpoint. In the SEPM workflow, a proposed revision passes validation and promotion checks before entering the active procedure pool; later similar tasks can retrieve the revised graph and warnings.

This expanded reading makes the memory–collaboration loop explicit: the procedure constrains coordination, the blackboard supplies evidence for corrections, and those verified corrections change what the team can remember and reuse.

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

The checked-in [`evaluation_matrix.json`](evaluation_matrix.json), [`manifests/`](manifests/), and [`scripts/run_paper_experiments.sh`](scripts/run_paper_experiments.sh) define the **local selected-case evaluation workflow** and its adapters. Full runs need the upstream benchmark repositories under `external/`, benchmark-specific services, and model credentials. The current selected-case matrix and ignored local result files do not by themselves regenerate the full benchmark scores shown above.

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

## 📄 Citation and License

A public citation record will be added when available. The local research draft is in `paper/SEPM.pdf` in this workspace; `paper/` is not part of the source release.

This code is licensed under [Apache 2.0](LICENSE).
