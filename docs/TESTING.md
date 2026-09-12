# Manual test plan

Run these in order. After each, the **Who did the work** note says whether that
step used Google Gemini, Tavily, or neither.

> **Free-tier budget.** Gemini allows ~20 requests **per day, per model**.
> One simple run = **3 Gemini calls** (plan, synthesize, critique) + **~4 Tavily
> searches**. A run where the loop fires = ~5 Gemini calls. So expect roughly
> **5-6 runs per day per model**. If you run out, change `GEMINI_MODEL` in
> `.env` to `gemini-3.5-flash` or `gemini-3.6-flash` - each model has its own
> separate daily quota. Tavily gives 1000 searches/month, which you will not
> exhaust.

---

## Test 1 - The graph draws itself

```bash
.venv/bin/python -m research_agent.cli --graph
```

**Expect:** a Mermaid diagram listing `plan`, `research`, `synthesize`,
`critique`, `human_review`, `finalize`, with two dotted arrows out of
`critique`.

**Who did the work:** *Neither.* No network at all. This is LangGraph reading
the graph you built in `graph.py`. If the dotted arrows are missing, the
conditional edge is not wired.

---

## Test 2 - The test suite

```bash
.venv/bin/pytest -v
```

**Expect:** 13 passed.

**Who did the work:** *Neither.* Every paid call is faked with `monkeypatch`.
This is why CI is free. It also means these tests cannot catch a real API
changing its response shape - that is what Test 3 is for.

---

## Test 3 - Both APIs respond

```bash
.venv/bin/python scripts/check_setup.py
```

**Expect:**

```
testing google ... WORKS  -> 'OK'
testing tavily ... WORKS  -> 3 result(s), first: 'What is LangGraph?'
```

**Who did the work:** *Both, minimally.* One tiny Gemini completion (asking it
to reply "OK") and one Tavily search. Costs ~1 Gemini call.

---

## Test 4 - A full run

```bash
.venv/bin/python -m research_agent.cli "What is a LangGraph reducer?"
```

**Expect** these lines, in order:

| Line | Who did the work |
|---|---|
| `[plan] plan=4 items` | **Gemini.** It read your question and invented 4 sub-questions. Nothing was searched yet. |
| `[research] findings=12 items` | **Tavily.** 4 searches x 3 results. Gemini is not involved. 12 = 4x3. |
| `[synthesize] draft=...` | **Gemini.** It was handed the 12 snippets and told to write only from them. |
| `[critique] critique=..., gaps=0` | **Gemini again**, in a different role - reviewer, not writer. |
| `DRAFT FOR REVIEW` | **Your code.** The graph paused at `interrupt()`. |

Type `approve` at the prompt.

| Line | Who did the work |
|---|---|
| `[human_review] human_feedback='approve'` | **You.** |
| `[finalize] final_report=...` | **Nobody.** On approval `finalize_node` returns the draft unchanged - no LLM call. Deliberate: approving should not cost money. |

**Check the citations are real.** Open two or three of the URLs at the bottom.
They should exist and be relevant. This is the honesty test - the agent is
instructed to use only what Tavily returned.

---

## Test 5 - The revision path

Run Test 4 again, but at the prompt type real notes instead of `approve`:

```
make it much shorter, just three bullet points
```

**Expect:** `[finalize]` produces a visibly different, shorter report.

**Who did the work:** **Gemini.** This time `finalize_node` took the non-approval
branch and made one more call to rewrite. Compare with Test 4, where approving
made zero calls.

---

## Test 6 - Durability (the best one)

Start a run, and when the `Type 'approve'...` prompt appears, press
**Ctrl+C**.

**Expect:**

```
Stopped without deciding. Nothing is lost - every step is checkpointed.
Resume any time with:

    python -m research_agent.cli --thread <some-uuid>
```

Copy that thread id. Now inspect what was saved:

```bash
.venv/bin/python scripts/inspect_thread.py <thread-id>
```

**Expect:** ~8 checkpoints, newest first, showing state accumulating one field
at a time - `question` then `plan` then `findings` then `draft`.

Now resume in a brand new process:

```bash
.venv/bin/python -m research_agent.cli --thread <thread-id>
```

**Expect:** `resuming a paused run - the research was already done`, then
straight to the draft. **No `[plan]`, no `[research]`, no `[synthesize]`.**

**Who did the work:** *Neither, until you approve.* Everything was read back
from `checkpoints.db` on disk. This is what "durable execution" means in the
roadmap PDF, and it is the single hardest thing to fake.

---

## Test 7 - The self-correction loop

The loop only fires when the critic is unhappy, so ask something broad:

```bash
.venv/bin/python -m research_agent.cli "Compare LangGraph, CrewAI and AutoGen for production multi-agent systems"
```

**Expect** `[research]` and `[synthesize]` to appear **twice**, with
`gaps=2 items` on the first `[critique]`.

**Who did the work:** **Gemini decided** there were gaps and wrote new search
queries; **Tavily ran** those new searches; **Gemini rewrote** the draft with
the extra material.

If the loop does not fire, the critic was satisfied - that is a pass, not a
failure. Force it by lowering the bar: try a vague question like
"tell me everything about AI agents".

---

## Test 8 - It fails loudly, not silently

```bash
TAVILY_API_KEY=deliberately-wrong .venv/bin/python -m research_agent.cli "What is LangGraph?"
```

**Expect:** an error mentioning `Tavily search failed: Error 401`.

**Who did the work:** **Gemini planned** (that call still succeeds), then
**Tavily rejected** the key.

This one matters. Tavily's library *returns* `{"error": ...}` instead of
raising, so an earlier version of `tools.py` read "no results" and the agent
politely reported it could not find anything - with a broken key and no
warning. Silent failure is the worst kind. Test 13 in the suite now locks this
behaviour in.

Your real key in `.env` is untouched - the bad value only exists for that one
command.

---

## Done

If 1-8 behave as described, the system is working end to end and it is time to
commit.
