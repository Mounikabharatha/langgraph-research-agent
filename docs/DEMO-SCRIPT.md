# Building the demo sessions

These four sessions are what an interviewer sees when they open the live app.
Between them they demonstrate every claim the README makes, and they cost
nothing to read once saved.

Run them **locally**, then commit the result (last section). Running them on
the hosted app is pointless - that filesystem is wiped when the app sleeps.

```bash
rm -f checkpoints.db memory.db          # start clean
.venv/bin/streamlit run app.py
```

Budget: roughly 20 Gemini calls. The six-model fallback gives you about 120
a day, so this is comfortable.

---

## 1. The self-correction loop — the headline feature

> **What causes hallucination in large language models, and which techniques
> have actually reduced it in production?**

Answer length **auto**, Max critic passes **2**.

Two-part questions usually leave the critic unsatisfied, so expect `research`
to run **twice**, ending at **2 critic passes** and ~24 sources.

Press **✅ Approve & close**.

**Why this question:** it is the one an interviewer is most likely to ask you
anyway, so the answer is worth reading on its own. And the subject is pointed
— an agent that cites its sources and re-checks its own draft is a direct
answer to the hallucination problem it is describing.

**What it proves:** the agent judged its own output, decided it was thin, and
went back to the web unprompted. Point at this first.

---

## 2. A revision — before and after, side by side

> **Compare RAG and fine-tuning for adapting an LLM to a company's internal
> documents**

When the draft appears, type into the box:

> *rather than long paragraphs give it to me in 5 short bullets*

Press **✏️ Request changes**, then **✅ Approve & close**.

**Why this question:** a genuine system-design interview question. The saved
session doubles as revision material for you.

**What it proves:** the session shows the original answer, your instruction
labelled *"You asked:"*, and the rewrite — with the superseded version
expanded beside it. Roughly 2,000 characters down to 600.

---

## 3. A follow-up chain — context carrying between runs

> **What is the Model Context Protocol?**

Approve it. Then, in the same session, type:

> *Which companies have adopted it?*

Set Answer length to **brief**, press **🔎 Research follow-up**, approve.

**Why this question:** MCP is recent enough that no model knows the adopter
list from training data, so the answer can only have come from a live search.
It also signals you follow the field.

**What it proves:** you wrote *"it"* and the agent knew what you meant,
because the previous report was handed to the planner. Open **Under the hood**
on the follow-up and the searches name MCP explicitly. The sidebar shows one
session marked **+1**, not two unrelated runs.

---

## 4. A narrow question in a different field

> **What is the current repo rate set by the Reserve Bank of India?**

Answer length **auto**. Approve.

**Why this question:** deliberately nothing to do with AI. Four AI questions
in a row invite the thought that the agent only handles one subject; this one
shows the domain comes from the question, not the code. It is also a number
that changes, so the answer can only have come from the live web.

**What it proves:** two or three sentences, not an essay — the writer earns
every extra paragraph.

---

### Swaps, if you prefer

- **1** → *"What are the trade-offs between vector search and keyword search
  for retrieval?"*
- **2** → *"Compare REST and GraphQL for a mobile backend"*
- **4** → *"What is the current stable release of Python?"*

Keep the shapes: one broad two-part question, one comparison you then shorten,
one chain with a pronoun, one narrow factual question outside AI.

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
