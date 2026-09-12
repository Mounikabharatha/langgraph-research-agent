# LangGraph Research Agent

A research assistant built with [LangGraph](https://langchain-ai.github.io/langgraph/).
Give it a question; it plans sub-questions, searches the web, writes a cited
answer, critiques its own work, loops back if the answer has gaps, and pauses
for human approval before finalising.

It is a small project on purpose. Every feature exists to demonstrate one
LangGraph capability that a plain LLM call cannot do.

## The graph

```mermaid
graph TD;
	__start__([__start__]):::first
	plan(plan)
	research(research)
	synthesize(synthesize)
	critique(critique)
	human_review(human_review)
	finalize(finalize)
	__end__([__end__]):::last
	__start__ --> plan;
	critique -.-> human_review;
	critique -.-> research;
	human_review --> finalize;
	plan --> research;
	research --> synthesize;
	synthesize --> critique;
	finalize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

The dotted edges out of `critique` are the conditional edge: if the critic
finds gaps and the revision budget is not spent, the agent loops back and
researches again. Otherwise it goes to human review.

## What each piece demonstrates

| Capability | Where it lives | Why it matters |
|---|---|---|
| Typed shared state | `state.py` | Every node reads and writes one validated `TypedDict` |
| State reducers | `Annotated[list, operator.add]` | Findings **accumulate** across loops instead of overwriting |
| Tool use | `tools.py` | The agent acts on the world, not just talks |
| Conditional edges / loops | `route_after_critique` | Self-correction — the actual point of a graph framework |
| Loop guard | `max_revisions` | Stops a picky critic from burning your API budget forever |
| Durable checkpointing | `SqliteSaver` in `graph.py` | Every step is saved; runs resume across processes |
| Human-in-the-loop | `interrupt()` in `nodes.py` | Graph pauses mid-run, a human decides, execution resumes |
| Observability | `tracing.py` | Every run named and tagged, with `thread_id` in metadata |
| Failure triage | `tools.py`, `research_node` | One dead query degrades the answer; *every* query failing stops the run |

## Background

This project started from a ChatGPT deep-research report on LangGraph and
agentic AI, kept here as
[docs/langgraph-roadmap-analysis.pdf](docs/langgraph-roadmap-analysis.pdf).

Two things worth knowing if you read it:

- **It describes a team-scale system.** Page 10 estimates 60–100 person-weeks
  across 3–6 developers. This repo implements the first of its three proposed
  architectures — "Monolithic (Single Agent Graph)" from page 6 — which the
  report itself recommends for a proof of concept with limited resources.
- **Its code samples do not run.** Several LangGraph APIs in it do not exist.
  See [A note on the LangGraph API](#a-note-on-the-langgraph-api) at the bottom
  for the corrections; this codebase uses the real API throughout.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # installs the package in editable mode
cp .env.example .env                  # then fill in your keys
```

You need two keys, both free:

- `GOOGLE_API_KEY` — https://aistudio.google.com/apikey (no credit card)
- `TAVILY_API_KEY` — https://tavily.com (1000 searches/month)

Then confirm both APIs actually respond:

```bash
python scripts/check_setup.py
```

### Choosing a model

Gemini's free tier is roughly **20 requests per day, per model**, and one run
costs 3–5 calls. Quota is tracked per model, so switching `GEMINI_MODEL` in
`.env` gives you a fresh allowance:

| Model | Trade-off |
|---|---|
| `gemini-3.1-flash-lite` (default) | Largest free allowance |
| `gemini-3.5-flash` | Better reasoning, smaller allowance |

To use OpenAI instead, set `LLM_PROVIDER=openai` and supply `OPENAI_API_KEY`.
Note the OpenAI API is billed separately from ChatGPT Plus.

## Tracing (optional)

Add these to `.env` and every node call shows up in
[LangSmith](https://smith.langchain.com) — prompt, response, token count and
latency, per step:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=research-agent
```

That is the whole setup. **Tracing is switched on by environment variables, not
by code** — there is no `invoke_with_observability()`, whatever generated
tutorials may claim.

What `tracing.py` adds on top is the part that makes traces navigable: each run
is given a `run_name`, tags, and — most usefully — the `thread_id` in its
metadata, so a trace can be matched to a row in `checkpoints.db`. Without that
every trace is called "LangGraph" and there is no way to tell which run it was.

`scripts/check_setup.py`, the CLI and the UI all print the current tracing
state, so you are never guessing whether it is on.

> `langsmith` caches its environment reads, so `.env` has to be loaded before
> the first graph call. Both entry points do this at startup.

## Run it

### Web UI

```bash
pip install -e ".[ui]"
streamlit run app.py
```

Opens at http://localhost:8501. Type a question, watch each node fire live,
then Approve or request changes. Past runs are listed in the sidebar — ⏸ marks
one paused at the review gate, ✓ one that finished. Clicking a paused run
reopens it exactly where it stopped, which is the checkpointer doing its job.

### Command line

```bash
python -m research_agent.cli "How do LangGraph checkpointers differ from LangChain memory?"
```

Print the graph diagram instead of running:

```bash
python -m research_agent.cli --graph
```

Press **Ctrl+C** at the approval prompt and the run pauses on disk. Resume it
later — in a different process — with no re-research:

```bash
python -m research_agent.cli --thread <thread-id>
```

Inspect what has been saved:

```bash
python scripts/list_threads.py              # every thread, finished or paused
python scripts/inspect_thread.py <id>       # step-by-step checkpoint history
```

## Tests

```bash
pytest
```

The tests run with **no API keys** — the LLM and search calls are faked — so CI
is free. They cover the routing logic, the loop guard, reducer accumulation,
and tool-failure handling.

Because they fake both APIs, they cannot catch a provider changing its response
shape. [docs/TESTING.md](docs/TESTING.md) is a manual plan that does, and it
labels every step with whether Gemini, Tavily, or neither did the work.

## Layout

```
src/research_agent/
├── state.py    # the shared State TypedDict + reducers
├── llm.py      # LLM construction (one place to swap providers)
├── tools.py    # web search
├── nodes.py    # the six nodes + the routing function  <- read this first
├── graph.py    # wiring and compilation
└── cli.py      # command-line entry point
tests/          # runs without API keys
```

## Roadmap

- [x] Plan → research → synthesize → critique loop
- [x] SQLite checkpointing and resumable threads
- [x] Human-in-the-loop approval gate
- [x] Tests + CI
- [x] Works on Google Gemini (free tier) or OpenAI
- [x] Streamlit front end
- [x] LangSmith tracing
- [ ] Postgres checkpointer for multi-user deployment
- [ ] Vector store for long-term memory across sessions

## A note on the LangGraph API

If you are working from a generated report or tutorial, be careful: several
commonly-hallucinated APIs do not exist. The correct forms, all used here:

| Frequently hallucinated | Actual API |
|---|---|
| `graph.create_runtime().invoke(s)` | `app = builder.compile()` then `app.invoke(s)` |
| `add_conditional_edge(...)` | `add_conditional_edges(source, router, path_map)` — plural |
| router returns a `bool` | router returns the **name of the next node** |
| `Runtime.invoke_with_observability()` | Set `LANGSMITH_TRACING=true` in the environment |
| `ToolNode(SomeTool())` | `ToolNode([tool1, tool2])` — takes a list |

Always check https://langchain-ai.github.io/langgraph/ before trusting a snippet.
