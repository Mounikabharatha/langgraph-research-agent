# Deploying the live demo

The app is a Streamlit server, so it needs a host that keeps a Python process
running with an open websocket.

**Vercel and Netlify cannot host it.** They run static sites and short-lived
serverless functions; there is nowhere for a long-running server to live, and
the live `plan → research → synthesize` streaming depends on that connection
staying open. Use one of these instead:

| Host | Cost | Notes |
|---|---|---|
| **Streamlit Community Cloud** | Free | Purpose-built. Deploys straight from GitHub |
| Hugging Face Spaces | Free | Also fine; pick the Streamlit SDK |
| Render / Railway | Free tier | General Python hosting, more configuration |

The steps below are for Streamlit Community Cloud.

## 1. Push everything first

```bash
git push origin main
```

The deploy reads your GitHub repo, so anything unpushed will not appear.

## 2. Create the app

1. Go to https://share.streamlit.io and sign in with GitHub
2. **New app** → pick `Mounikabharatha/langgraph-research-agent`
3. Branch `main`, main file path `app.py`
4. **Deploy**

First build takes a few minutes while it installs the dependencies.

## 3. Add your keys

The repo has no keys in it, by design — the app will start but cannot do
anything until you supply them.

In the app's **⋮ → Settings → Secrets**, paste:

```toml
GOOGLE_API_KEY = "your-key"
TAVILY_API_KEY = "your-key"
LLM_PROVIDER = "google"
GEMINI_MODEL = "gemini-3.1-flash-lite"
```

`app.py` copies anything in Secrets into the environment at startup, so the
rest of the code keeps reading `os.getenv` and never learns about Streamlit.

Save, and the app restarts with the keys in place.

## 4. Your link

```
https://<app-name>.streamlit.app
```

That is what goes on your CV and in the README.

## What to expect from the free tier

**Storage is ephemeral.** Streamlit Community Cloud wipes the filesystem when
the app sleeps or redeploys, so `checkpoints.db` and `memory.db` start empty
again. Saved sessions do not survive a restart. If you want them to, swap
`SqliteSaver` for `langgraph-checkpoint-postgres` pointed at a hosted Postgres
such as Supabase - the change is contained to `graph.py` and `memory.py`.

**Model quota is the real limit.** Gemini's free tier allows roughly 20
requests per day *per model*, and one question costs 3-5. `llm.py` therefore
tries six models in turn, failing over when one is exhausted, which lifts the
practical ceiling from about five questions a day to roughly twenty-five.
When every model is spent the app says so plainly instead of showing a stack
trace, and points the visitor at the saved sessions.

Google retires models without much warning - `gemini-2.5-flash` was listed by
the API and returned 404 when called. If answers start failing, run:

```bash
python scripts/check_models.py
```

It calls each model in the chain with the structured-output request the
planner needs, and tells you which are working, exhausted or broken.

**The app sleeps when idle** and takes 30-60 seconds to wake. Left alone that
cold start lands on whoever opens the link from a CV, so
`.github/workflows/keep-warm.yml` visits the app every hour to keep the
container up, which brings a click down to a few seconds.

It has to use a real browser. Requesting the URL with curl returns the
hosting shell from share.streamlit.io, which renders the app in an iframe at
`/~/+/`; the container itself only starts when a client opens the websocket.
So an uptime pinger pointed at the URL looks like it is working while the app
goes on sleeping.

Each run logs how long the load took. If one reports a cold start, the hourly
schedule is not keeping up and the cron needs tightening. Note that GitHub
disables scheduled workflows after 60 days with no commits to the repo - if
the demo starts feeling slow again months from now, check that first.

## Before you share the link

Ask three or four good questions and approve them, so a visitor arriving to an
exhausted quota still has real sessions to read - complete with sources,
critic passes and revision history. Those cost nothing to browse.
