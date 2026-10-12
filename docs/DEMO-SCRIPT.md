# Building the demo sessions

This is what an interviewer sees when they open the live app, and it costs
them nothing to read.

Two sessions. The first is a five-question chain on one topic that exercises
every control in the app; the second is a single short question in a
different field, so the set does not imply the agent only handles one
subject.

Run them **locally**, then commit the result (last section). Running them on
the hosted app is pointless — that filesystem is wiped when the app sleeps.

```bash
rm -f checkpoints.db memory.db          # start clean
.venv/bin/streamlit run app.py
```

Budget: roughly 25 Gemini calls. The six-model fallback gives you about 120
a day, so this is comfortable.

---

# Session 1 — one topic, five questions, every control

Topic: **Retrieval-Augmented Generation**. A real system-design interview
subject, and it has natural follow-ups, which is what makes it a good chain.

Do all of this **without leaving the session**. Only the very last step closes
it.

## Q1 · the opening question — auto, and the critic loop

Answer length **auto**, Max critic passes **2**.

> **What is Retrieval-Augmented Generation and when should a team use it
> instead of fine-tuning?**

Two-part questions usually leave the critic unsatisfied, so expect `research`
to run **twice** and the run to end at **2 critic passes**, ~24 sources.

### Then revise it, before going anywhere

In the box at the review gate, type:

> *cut the background and keep only what a team needs to decide, as 5 bullets*

Press **✏️ Request changes**.

**Watch for:** the previous answer stays on screen, expanded, under *"replaced
because you asked for a change"*, with your instruction labelled **"You
asked:"** between the two. Roughly 2,000 characters down to 600.

> This is the single most useful thing to point at. It shows the agent
> rewrote to order **without searching again** — the sources are unchanged.

## Q2 · a follow-up — brief, and a pronoun

Still at the gate. Set Answer length to **brief** and type:

> **What are its main failure modes?**

Press **🔎 Research follow-up**.

**Watch for:** you wrote *"its"* and never said RAG. Open **Under the hood**
on the new answer — the planned searches name retrieval-augmented generation
explicitly, because the previous report was handed to the planner. The answer
itself is three sentences, because you asked for brief.

## Q3 · a follow-up — detailed

Set Answer length to **detailed** and type:

> **How do hybrid search and reranking fix those failures?**

Press **🔎 Research follow-up**.

**Watch for:** headings and several paragraphs. Put side by side with Q2 this
shows the length control is doing real work, not decoration.

## Q4 · a follow-up — back to auto

Set Answer length back to **auto** and type:

> **How would I measure whether my RAG pipeline is actually working?**

Press **🔎 Research follow-up**.

**Watch for:** a middle-sized answer. Auto reads the question and lands
between Q2 and Q3 without being told.

## Q5 · a follow-up — brief, and live data

Set Answer length to **brief** and type:

> **Which vector database is most used in production right now?**

Press **🔎 Research follow-up**, then **✅ Approve & close**.

**Watch for:** a short, current answer with live sources. "Right now" is the
point — this cannot have come from training data.

## What the finished session looks like

The sidebar holds **one** entry marked **+4**. Opening it shows all five
questions in order with their answers, the revision history on Q1, and
*Under the hood* on each with its own sub-questions, sources and critic
passes.

Every claim in the README is visible in that one session:

| Claim | Where to point |
|---|---|
| Plans its own sub-questions | *Under the hood* on any question |
| Real, clickable sources | *Sources* on any question |
| Critiques itself and retries | Q1, 2 critic passes, ~24 sources |
| Rewrites to order without re-searching | Q1's revision, both versions shown |
| Carries context between runs | Q2 — *"its"* resolved to RAG |
| Length matches the request | Q2 brief vs Q3 detailed vs Q4 auto |
| Searches the live web | Q5 — "right now" |
| Human approves before anything closes | the whole session waited for you |

---

# Session 2 — a second chain, in a completely different field

Topic: **the RBI repo rate**. Nothing to do with AI, which is the point — five
AI questions in a row invite the thought that the agent only handles one
subject. It is also a number that changes, so every answer has to come from
the live web.

Again, do not leave the session until the last step.

## Q1 · the opening question — auto *(already done)*

> **What is the current repo rate set by the Reserve Bank of India?**

**2 critic passes, 9 sources.** Worth reading before you go on: the first
draft said 5.25%, the critic flagged it as outdated, and the final answer
reports the disagreement rather than picking one —

> *Sources from August 2026 cite 5.25%, while a source dated 7 October 2026
> reports 5.50%.*

**This is the best single thing in either session.** Most tools state one
number confidently. Yours found the sources conflicted, noticed the dates and
said so, with citations for both. If an interviewer reads one answer, make it
this one.

## Q2 · a follow-up — brief, and a pronoun *(waiting at the gate)*

> **When was it last changed?**

Already run at **brief**, 2 critic passes, 12 sources. You wrote *"it"* and
never said "repo rate" — open **Under the hood** and the searches name it.

### Revise it before moving on

In the box, type:

> *just the date and the new rate, one line*

Press **✏️ Request changes**.

**Watch for:** the previous answer stays visible above, labelled *replaced
because you asked for a change*, with your instruction between the two — and
the sources are identical, because a revision does not search again.

## Q3 · a follow-up — detailed

Set Answer length to **detailed**:

> **How does a change in the repo rate affect home loan EMIs in India?**

**🔎 Research follow-up**

**Watch for:** headings and worked detail. Next to Q2's one-liner, this is the
length control visibly doing work.

## Q4 · a follow-up — auto, and a definition

Set Answer length to **auto**:

> **What is the difference between the repo rate and the reverse repo rate?**

**🔎 Research follow-up**

**Watch for:** it should *define the difference*, not re-answer what the
current rate is. Earlier versions drifted back to the parent topic on a
question like this; the planner is now told to use the earlier answer only to
resolve what the question refers to.

## Q5 · a follow-up — brief, and wider context

Set Answer length to **brief**:

> **Which other major central banks have changed rates in 2026?**

**🔎 Research follow-up**

**Watch for:** a short answer that leaves India behind entirely — the chain
carries context without being trapped by it.

## Q6 · a follow-up — auto, and a future date

Set Answer length to **auto**:

> **When is the next RBI monetary policy committee meeting?**

**🔎 Research follow-up**, then **✅ Approve & close**.

**Watch for:** a date in the future. No model knows this from training data,
so it can only have come from the live search.

## What the finished session looks like

A second sidebar entry marked **+5**, on a subject with no overlap with the
first. Between the two sessions an interviewer can see every control used,
on two unrelated topics, without spending anything to read them.

## Saving them into the repo

```bash
mkdir -p demo
cp checkpoints.db memory.db demo/
git add -f demo/checkpoints.db demo/memory.db
git commit -m "chore: refresh the demo sessions"
git push origin main
```

`seed_demo_data()` in `graph.py` copies these into place at startup when no
database exists, so every fresh deploy comes up with the sessions already
there. An existing database is never overwritten, so this does nothing on a
machine you have been using.

Re-run this whenever the sessions get stale - question 4 in particular ages,
since the AQI it quotes is from the day you ran it.
