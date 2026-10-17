# Science Agent

An LLM-agnostic agent that triages scientific ideas from first principles: is an idea
**wrong**, already **known**, **plausible but hard**, or **promising** enough to research further?

It is built for questions like *"Can I enhance the Shannon theorem by ...?"* or
*"What about higher dimensions for transmitting data?"*, and answers with a written report that
names the deciding argument, the relevant physical limits, a computed number, an adversarial
critique, and the cheapest next step.

It runs with **any model**: local models via Ollama, LM Studio, vLLM or llama.cpp; hosted models
via any OpenAI-compatible API; or Claude via the Anthropic SDK. It can also run inside
Claude Code as a skill (see [Claude Code skill](#claude-code-skill)).

## How it works

```
idea ─► 1 formalize ─► 2 literature ─► 3 limits ─► 4 compute ─► 5 critic ─► 6 verdict ─► report
         precise        arXiv +          check vs     LLM-written   adversarial   wrong / known /
         claim,         Semantic         known        Python run    review in     plausible_but_hard /
         assumptions    Scholar          limits       in sandbox    own prompt    promising
```

| Stage | What it does |
|---|---|
| **formalize** | Restates the idea as a testable claim with assumptions, variables and standard terminology. Stops with clarifying questions if the idea is too vague. |
| **literature** | Searches arXiv and Semantic Scholar, then maps the idea to its established field and judges novelty. |
| **limits** | Checks the claim against a curated knowledge base of fundamental limits. The key question: *which assumption of the limit does the idea relax?* If none, it cannot beat the limit. |
| **compute** | Has the model write a short NumPy/SciPy/SymPy script for the most decisive number, runs it in a subprocess, and retries once on errors. |
| **critic** | A separate adversarial prompt looks for hidden assumptions, renamed existing techniques and errors in the earlier stages. You can give it a stronger model than the other stages. |
| **verdict** | Weighs everything and gives one verdict, a confidence, the deciding argument and a next step. |

Design choices:

- **Control flow in code, not in the model.** Each stage is one prompt that returns JSON. No
  native tool calling is needed, which keeps small local models usable.
- **Robust to sloppy output.** The JSON extractor handles code fences, surrounding prose,
  trailing commas and `<think>` blocks; invalid output is re-asked once with the error.
- **Calibration over enthusiasm.** Language models tend to praise ideas. The system prompt,
  the separate critic and the [benchmark](#calibration-benchmark) all push against that.

## Install

```bash
git clone https://github.com/theoalm66/agent-science
cd agent-science
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ,anthropic for the Claude backend
cp config.example.yaml config.yaml # then edit
```

## Configure a model

`config.yaml` (git-ignored). Local model with Ollama:

```yaml
llm:
  provider: openai_compatible
  base_url: http://localhost:11434/v1
  model: qwen3:32b
```

Other OpenAI-compatible servers only need a different `base_url` (LM Studio
`http://localhost:1234/v1`, vLLM `http://localhost:8000/v1`) and optionally `api_key_env`.

Claude:

```yaml
llm:
  provider: anthropic
  model: claude-opus-5-5
  effort: high
```

Stages can use different models. A strong critic is the most effective single upgrade when the
rest runs on a small local model:

```yaml
stages:
  critic: {provider: anthropic, model: claude-opus-5-5}
```

You can also override from the command line: `--provider`, `--model`, `--base-url`.

## Use

```bash
science-agent assess "A fractal modulation that beats B*log2(1+SNR) on a single-antenna AWGN channel"
science-agent assess --file my_idea.txt
science-agent assess "..." --no-literature --no-compute   # faster, offline

science-agent search "orbital angular momentum multiplexing capacity"
science-agent limits information_theory                   # print the knowledge base
```

Each run writes `reports/<date>-<slug>.md` (readable report) and a `.json` trace with every
stage's output.

## Calibration benchmark

`benchmarks/information_theory.yaml` holds 12 ideas with established answers: 7 that are wrong
(e.g. compressing any file, faster-than-light signalling with entanglement, unlimited OAM capacity,
beating the Holevo bound) and 5 that are real (MIMO, superdense coding, polarization multiplexing,
semantic communication, fibre nonlinearity compensation).

```bash
science-agent bench
science-agent bench --only oam_unlimited,mimo_scaling
```

The score reports both failure modes separately: wrong ideas the agent let through, and real
ideas it wrongly rejected. Run it whenever you change a model, prompt or knowledge base. An
assessor that calls everything wrong scores just as badly as one that calls everything promising.

## Adding a domain

Copy `src/science_agent/knowledge/information_theory.yaml` to `<domain>.yaml` and list the domain's
fundamental limits. For each limit, give its **assumptions** and the **legitimate workarounds**
that relax exactly one assumption; that structure is what lets the agent tell "beats the limit"
from "changes the problem". Then add a benchmark file with known answers and run
`science-agent bench benchmarks/<domain>.yaml`. You can also pass `--domain path/to/file.yaml`.

## Claude Code skill

The repository also ships the same pipeline as a Claude Code skill: open this repo in Claude Code
and run `/assess-idea <your idea>`. It reuses the prompts and knowledge base from
`src/science_agent/`, and delegates to two subagents in `.claude/agents/`:
`literature-scout` (search) and `critic` (adversarial review in a fresh context). Reports go to
`reports/` in the same format, so the two routes can be compared.

## Safety note on the compute stage

The compute stage executes model-written Python in a subprocess with a time limit, a 4 GB memory
limit and an empty working directory. That contains accidents, not attacks: the script runs as
your user and can reach the network and your files. For untrusted models, run the agent in a
container, or use `--no-compute`.

## Limitations

- This is a first-pass triage, not peer review. Check the deciding argument yourself.
- Small local models make more reasoning errors; measure them with the benchmark before trusting them.
- Literature search covers arXiv and Semantic Scholar only. Without `S2_API_KEY`, Semantic
  Scholar rate limits are tight.

## Development

```bash
pytest          # offline; uses a scripted fake LLM and mocked HTTP
```
