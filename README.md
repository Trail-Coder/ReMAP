# ReMAP



## Relation-Deferred Memory Poisoning Evaluation Framework



ReMAP is a modular research framework for studying indirect prompt injection and long-term memory poisoning in tool-using LLM agents. The implementation focuses on a relation-deferred attack construction: instead of placing a complete malicious policy in one document, the attack produces independently plausible evidence records whose semantic relation is induced only when the records are jointly processed during memory construction.



The repository is organized as an extensible attack--defense--agent--dataset evaluation pipeline. Each component is discovered dynamically, so new attacks, defenses, agents, scorers, and datasets can be added without rewriting the experiment orchestration layer.



> **Research-use notice.** This code is intended for controlled security evaluation, defensive research, and reproducibility studies. Do not deploy the attack components against systems or data for which you do not have explicit authorization.



## Current repository scope



Due to the size limitation of Code and Data Supplement, the working copy has been deliberately reduced to the Memory Injection evaluation path. The active dataset registry contains one dataset:



- **Memory Injection** — the released document/query/goal benchmark used by the ReMAP memory-poisoning evaluation.



Other benchmark families (agent, non-English, smoke, and auxiliary ablations) and their generic loaders have been removed to reduce repository size. The retained tree contains only the Memory Injection behavior split and its corresponding behavior-only preprocessing artifacts.





## System architecture



```text

                         ┌──────────────────────┐

                         │  Dataset loader      │

                         │  Sample(document,    │

                         │  query, metadata)    │

                         └──────────┬───────────┘

                                    │

                                    ▼

                         ┌──────────────────────┐

                         │ Attack transformation│

                         │  ReMAP: e_C + e_A    │

                         └──────────┬───────────┘

                                    │

                                    ▼

                         ┌──────────────────────┐

                         │ Per-record defense   │

                         │ block or clean each  │

                         │ external record      │

                         └──────────┬───────────┘

                                    │

                                    ▼

       ┌──────────────────────┐   ┌──────────────────────┐

       │ Scoped long-term     │──▶│ Structured Agent     │

       │ memory retrieval     │   │ context construction │

       └──────────────────────┘   └──────────┬───────────┘

                                             │

                                             ▼

                                  ┌──────────────────────┐

                                  │ Tool-using Agent     │

                                  │ + SQLite memory write│

                                  └──────────┬───────────┘

                                             │

                                             ▼

                                  ┌──────────────────────┐

                                  │ A-ASR / M-ASR / B-ASR│

                                  │ scoring and reports  │

                                  └──────────────────────┘

```



The main orchestration logic is implemented in [framework/runner.py](framework/runner.py). The framework keeps the following boundaries explicit:


1. Dataset content is represented as a `Sample` object.

2. Attack transformations produce one or more external records.

3. Each external record is screened independently by the defense.

4. Only records that pass screening enter the Agent context.

5. Long-term memory is isolated by attack/defense/agent/dataset combination.

6. The Agent result is scored after the memory state has been updated.



## Repository layout



```text

.

├── agents/                         Agent implementations and tool loops

├── attacks/                        Attack adapters, including ReMAP

├── datasets/                       Active dataset adapters

│   ├── __init__.py

│   └── sleeper_benchmark.py        Memory Injection loader

├── defenses/                       Input and system-prompt defenses

├── framework/                      Core types, registry, and Runner

├── scorers/                        Lightweight scoring integrations

├── tools/                          Evidence generation and semantic Judge tools

├── Baselines/Sleeper_adapted/      Retained Memory Injection-related adapter

├── memory_store.py                 Scoped SQLite long-term memory

├── remap_core.py                   CAS and anchor optimization

├── run_experiment.py               Main command-line entry point

├── config.py                           API configuration loader

└── api.json.example                API configuration template

```



Large raw datasets, non-Memory-Injection benchmark trees, generated result files, and runtime caches are intentionally excluded from the working copy.



## ReMAP attack pipeline



The active attack implementation is [attacks/remap.py](attacks/remap.py). It follows a relation-deferred construction:


```text

Malicious policy = (Scenario C, Relation R, Action A)



e_C: scenario evidence containing a shared anchor

e_A: action/capability evidence containing the same anchor

R:    intentionally omitted from either record

```



The generation path is:



1. Generate multiple `(scenario, action, anchor)` candidates with few-shot prompting.

2. Reject candidates that contain explicit tool names or fail the shared-anchor constraint.

3. Rank candidates with Cross-Evidence Attention Score (CAS) when the surrogate model is available.

4. Optionally perform anchor beam optimization.

5. Return two independently screened records through `apply_records()`.



If external generation or the local surrogate model is unavailable, ReMAP falls back to conservative built-in evidence templates. This permits structural testing, but it is not equivalent to a full paper-scale generation run.



## Dataset contract and preprocessing



The retained loader is [datasets/sleeper_benchmark.py](datasets/sleeper_benchmark.py). It converts each released record into:



```python

Sample(

    document=document["text"],

    query=query["query"],

    sample_id=f"{domain}_{doc_id}",

    metadata={

        "domain": ...,

        "format": ...,

        "goal_text": ...,

        "goal_category": ...,

        "query_category": ...,

        "query_style": ...,

        "preexisting_memories": ...,

    },

)

```



The loader supports:



- deterministic subsampling with `n` and `seed`;

- optional filtering by document domain;

- preservation of pre-existing memories supplied by the released benchmark;

- retention of goal and query annotations for scoring and audit context.



The current Runner does not reinterpret the dataset's goal as a new user instruction. The goal is retained as metadata and passed to the optional semantic judge as attack context.



## Long-term memory model



[memory_store.py](memory_store.py) provides a lightweight SQLite backend. The database is created lazily at:



```text

data/agent_memory.db

```



Memory behavior is scoped as follows:



- a new scope is created for each defense/agent/attack/dataset combination;

- memory is cleared at the beginning of each combination;

- memories persist across samples within that combination;

- retrieval uses lexical overlap with the current query and falls back to recency;

- Agent `save_memory` calls write actual records to SQLite rather than being no-op simulations.



This implementation is intentionally simple and transparent. It is suitable for controlled ablations and pipeline verification, but it is not a production vector database or a replacement for an agent's proprietary memory subsystem.



## Available components



Components are discovered by [framework/registry.py](framework/registry.py).


### Attack



- `ReMAP`



### Dataset



- `Memory Injection`



### Defenses



The current tree contains the following defense implementations, subject to their optional dependencies and configuration:



- `DataFilter`

- `PromptArmor`

- `GEPA Hardening`

- `IntentGuard`

- `Prompt Hardening v3`

- `Prompt Hardening`

- `Untrusted Markers`



### Agents



The current tree contains:



- `Native ReAct`

- `Constitutional`

- `CoT-Guard`

- `Reflexion`

- `Self-Refine`

- `Two-Agent-Verify`

- `AgentPro`



Some optional agents require external packages that are not part of the minimal environment. Missing optional dependencies are reported during discovery and do not necessarily prevent the remaining components from being listed.



## Environment setup



The repository does not currently provide a single complete `requirements.txt` for every optional component. The minimal orchestration path requires Python 3.9+ and an HTTP client:


```powershell

python -m pip install requests

```



Additional packages may be needed depending on the selected components:



- `torch` and `transformers` for CAS computation and anchor optimization;

- the corresponding AgentPro/OpenClaw packages for those agents;

- model-provider-specific packages if a custom Agent implementation requires them.



The code uses an OpenAI-compatible `/chat/completions` endpoint. Configure it either with a root `api.json` file based on [api.json.example](api.json.example), or through environment variables:


```powershell

$env:OPENAI_API_KEY  = 'your-api-key'

$env:OPENAI_BASE_URL = ''

$env:OPENAI_MODEL    = 'gpt-4o'

```





## Basic usage



First inspect the components that can be imported in the current environment:



```powershell

python -B run_experiment.py --list

```



A minimal Memory Injection run can be launched with:



```powershell

python run_experiment.py `

  --datasets 'Memory Injection' `

  --attacks ReMAP `

  --defenses 'Untrusted Markers' `

  --agents 'Native ReAct' `

  --n 10 `

  --seed 42 `

  --output results

```



The equivalent single-line command is:



```powershell

python run_experiment.py --datasets "Memory Injection" --attacks ReMAP --defenses "Untrusted Markers" --agents "Native ReAct" --n 10 --seed 42 --output results

```



Useful options:



```text

--n N                    number of samples per combination

--seed SEED              deterministic dataset sampling seed

--max-agent-iterations K maximum Agent tool-loop iterations

--attacks A1,A2          restrict attack implementations

--defenses D1,D2         restrict defenses

--agents G1,G2           restrict Agents

--datasets D1,D2         restrict datasets

--output PATH            result directory

--semantic-judge         enable optional LLM semantic scoring

```



The semantic judge makes additional model API calls and is disabled by default. It should only be enabled when the API configuration and evaluation budget are available.



## Generated outputs



When an experiment is executed, the Runner creates the requested output directory and writes:



```text

results/

├── experiment_partial.json

├── experiment_results.json

└── experiment_report.md

```



The partial JSON is updated after each completed combination. The final JSON contains the configuration and per-combination aggregates. The Markdown report contains pass rates, attack success summaries, and detailed counts.



These files are generated artifacts and are intentionally not part of the source distribution.



## Metrics



The paper-facing terminology is:



- **A-ASR** — action attack success rate: the document caused the Agent to execute the target action that was not requested by the user;

- **M-ASR** — memory attack success rate: the document caused the Agent to write the malicious policy or its induced relation to long-term memory;

- **B-ASR** — both attack success rate: A-ASR and M-ASR are simultaneously true;

- **Pass Rate** — fraction of samples that were not blocked by the defense;

- **Block Rate** — fraction of samples rejected before memory construction;

- **Clean Rate** — fraction of passed samples whose external records were modified by the defense.



The default scorer uses observable tool calls and verifies that a `save_memory` call produced a persisted SQLite write. The optional semantic scorer evaluates the complete interaction trace with an LLM and can be enabled with `--semantic-judge`.



## Reproducibility and known limitations



The following limitations should be considered before interpreting results:



1. The upstream release omitted raw datasets because of file-size restrictions; this working copy retains only the active behavior split.

2. The default runtime does not guarantee that every optional Agent dependency is installed.

3. ReMAP evidence generation may fall back to built-in templates if the generation API or CAS surrogate is unavailable.

4. The current SQLite retrieval implementation uses lexical overlap rather than embedding retrieval.

5. Tool effects such as reference verification are simulated for evaluation; this framework is not an operational document-verification service.

6. Full paper-scale experiments require compatible model endpoints, the local surrogate model, and sufficient compute/API budget.







## Troubleshooting



### `FileNotFoundError` when loading Memory Injection



The loader first checks the retained local file under `datasets/LLM-agent-memory-poisoning`. If the data is stored elsewhere, set `MEMORY_INJECTION_DATA_PATH` to the absolute JSON path.



### No Agent appears in the registry



Run `python -B run_experiment.py --list`. Optional Agent packages may be missing; use an Agent whose dependencies are installed.



### ReMAP falls back to templates



Check the API configuration and, for CAS/beam optimization, install `torch` and `transformers` and set `REMAP_SURROGATE_MODEL` to a valid local model directory.



### Results from a previous experiment are unexpectedly present



The Runner regenerates `experiment_partial.json`, `experiment_results.json`, and `experiment_report.md` under `--output`. These are runtime artifacts and can be removed after archiving any required measurements.



## License and attribution



The repository combines project-specific orchestration code with selected baseline and defense implementations. Please consult the license files within the corresponding subdirectories before redistribution. Dataset licenses and citation requirements are governed by the original Memory Injection release and must be respected independently of this wrapper repository.

