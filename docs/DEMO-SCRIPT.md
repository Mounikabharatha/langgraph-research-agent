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

> **What are the main causes of air pollution in Delhi, and which measures
> have actually reduced it?**

Leave Answer length on **auto**, Max critic passes on **2**.

Two-part questions usually leave the critic unsatisfied, so you should see
`research` run **twice** and end with **2 critic passes** and ~24 sources.

Press **✅ Approve & close**.

**What it proves:** the agent judged its own draft, decided it was thin, and
went back to the web unprompted. This is the thing that makes it an agent
rather than a chatbot, and it is the first thing to point at.

---

## 2. A revision — before and after, side by side

> **Compare electric and hybrid cars for Indian drivers**

When the draft appears, type into the box:

> *rather than long paragraphs give it to me in 5 short bullets*

Press **✏️ Request changes**. Then **✅ Approve & close**.

**What it proves:** the session now shows the original answer, your
instruction labelled *"You asked:"*, and the rewritten version - with the
superseded one expanded so the two can be compared. Roughly 2,000 characters
down to 600.

---

## 3. A follow-up chain — context carrying between runs

> **Which laptop is best for gaming in 2026?**

Approve it. Then in the same session, type into the box:

> *What does it cost in India?*

Set Answer length to **brief**, and press **🔎 Research follow-up**.
Approve that too.

**What it proves:** you wrote *"it"* and the agent knew which laptop, because
the previous report was handed to the planner. Open **Under the hood** on the
follow-up and the searches name the actual model. The sidebar shows one
session marked **+1**, not two unrelated runs.

---

## 4. A narrow question — length matching the question

> **What is the current AQI in Delhi?**

Answer length **auto**. Approve.

**What it proves:** two or three sentences, not an essay - the writer is told
to earn every extra paragraph. It also proves the web search is real: no
model knows today's AQI from training data.

---

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
