# Multi-Level Decision-Making Agent

DSPy-powered, LangGraph-orchestrated FastAPI service for law-firm client comms:
triage, risk, sentiment, event detection, contextual replies, micro/high-level
insights, and outbound message drafting.

## Architecture

```
                  +----------------------+
   client req  ->  |  FastAPI (uvicorn)  |
                   |  slowapi rate-limit |
                   |  structlog access   |
                   +----------+----------+
                              |
        +---------------------+---------------------+
        v                     v                     v
  /api/v1/chat         /api/v1/insights      /api/v1/outbound
   + /agent
        |                     |                     |
        v                     v                     v
 +-----------------+   +-----------------+   +-----------------+
 | ChatOrchestr.   |   | MicroInsight    |   | Outbound        |
 |   |             |   | + HighLevel     |   | Generator       |
 |   v             |   | (DSPy)          |   | (DSPy)          |
 | chat_graph      |   +-----------------+   +-----------------+
 | (LangGraph DAG) |
 +--+--------------+
    |
    v
            START
              |
       retrieve_context           <-- prior insights / case state
              |
            triage
            /  |  \
sentiment   event   risk          <-- parallel BSP step
            \  |  /
            decide
              |
   +----------+----------+
   v          v          v
 skip      respond   await_human  <-- FLAG+High pauses via interrupt()
   |         |          |
   |       critic       | (resume with Command(resume={action,reply,reviewer}))
   |       /   \        |
   |  refine   done     |
   |    |      |        |
   |  respond  |        |
   |  (loop)   |        |
   |    |      |        |
   |    +-(<MAX)         |
   |                     |
   v                     v
              END
```

Every LM call is a tiny `dspy.Module` wrapping a `dspy.Signature`. Tier
selection (`main` / `fast` / `summary` / `report`) lives in
`app.core.llm`, so swapping providers (OpenAI / Gemini / anything
LiteLLM-compatible) is a single env-var change per tier.

### Why LangGraph over LangChain (LCEL)

The graph isn't here for the parallel branches — LCEL's `RunnableParallel`
could do that. It's here for the things LCEL *can't* do:

| Feature                                | LCEL                | LangGraph                                 |
| -------------------------------------- | ------------------- | ----------------------------------------- |
| Linear chains                          | great               | overkill                                  |
| Parallel branches with merge           | `RunnableParallel`  | works, no advantage                       |
| **Loops with state-driven exit**       | not native          | `add_conditional_edges` back to a node    |
| **Human-in-the-loop interrupts**       | none                | `interrupt()` + `Command(resume=...)`     |
| **Resumable / persistent state**       | stateless           | checkpointers (Memory / Sqlite / Postgres)|
| **Per-thread conversation memory**     | bring your own      | `thread_id` config out of the box         |
| **Streaming intermediate state**       | token-level only    | per-node updates via `astream`            |

Where each one shows up in this graph:

- **Loop** — `respond` -> `critic` -> (refine) -> `respond`, bounded by
  `MAX_REFINE`. The critic scores the draft on tone / safety / brevity
  and feeds notes back into the next pass.
- **Interrupt** — `await_human` for FLAG + High risk. The graph pauses,
  the API returns `{event: "paused", thread_id, next: ["await_human"]}`;
  resumption with `POST /agent/resume` carries the reviewer's decision.
- **Checkpointer** — `MemorySaver` by default, swap for
  `AsyncSqliteSaver` or a Postgres saver in prod. The same `thread_id`
  picks up exactly where it left off (paused or otherwise).
- **Streaming** — `/agent/stream` emits one SSE event per node update,
  letting the client render triage first, then sentiment / event / risk
  arriving together, then respond / critic / refine.

## Layout

```
app/
  core/                 # config, llm provider, logging, rate limit, exceptions
  features/
    chat/               # signatures + DSPy modules + routes (analyze, summarize, concise)
    insights/           # micro + high-level (Gemini Pro by default)
    outbound/           # check-in / follow-up / appointment / case update
    agent/              # chat_graph (LangGraph DAG) + SSE /stream + /resume
  shared/               # request/response schemas, sanitizers, response envelope
  main.py               # app factory
deploy/
  k8s/
    base/               # deployment, service, configmap, secret, hpa, pdb, networkpolicy
    overlays/           # dev / staging / prod (kustomize)
  gcp/                  # cloudbuild, workload-identity, cloud-sql-proxy, external-secrets
tests/
```

## Quickstart

```bash
cp .env.example .env       # set OPENAI_API_KEY (and GEMINI_API_KEY if using report tier)
make install               # uv preferred, falls back to pip
make dev                   # uvicorn --reload on :8000
```

Or with the full stack:

```bash
docker compose up -d       # api + postgres + redis
curl http://localhost:8000/health
```

OpenAPI docs: http://localhost:8000/api/docs

## API surface

| Method | Path                                  | What it does                          |
| ------ | ------------------------------------- | ------------------------------------- |
| POST   | `/api/v1/chat/analyze`                | Full triage + risk + sentiment + reply|
| POST   | `/api/v1/chat/summarize`              | Conversation summary                  |
| POST   | `/api/v1/chat/make-concise`           | Shorten text (<= 4 words)             |
| POST   | `/api/v1/insights/micro`              | Per-client one-sentence insight       |
| POST   | `/api/v1/insights/high-level`         | Firm-wide leadership report           |
| POST   | `/api/v1/insights/summary`            | Dashboard insights JSON               |
| POST   | `/api/v1/outbound/generate`           | Weekly check-in draft                 |
| POST   | `/api/v1/outbound/follow-up`          | Follow-up message                     |
| POST   | `/api/v1/outbound/appointment-reminder` | Reminder draft                      |
| POST   | `/api/v1/outbound/case-update`        | Case progress message                 |
| POST   | `/api/v1/agent/stream`                | SSE stream of chat_graph node updates |
| POST   | `/api/v1/agent/resume`                | Resume a paused thread with reviewer's decision |

## Configuration

All settings come from env vars (see `.env.example`). Highlights:

- `LM_MAIN`, `LM_FAST`, `LM_SUMMARY`, `LM_REPORT` — `provider/model` strings;
  DSPy + LiteLLM route based on the prefix.
- `LOG_JSON` — `true` in prod, `false` for human-readable dev logs.
- `RATE_LIMIT` — slowapi-style string (e.g. `100/minute`); honored globally.
- `REDIS_URL` — when set, slowapi uses it as the rate-limit backing store.

## Deployment

Local image:

```bash
make build                       # docker build -t mldm-agent:dev
```

Kubernetes (kustomize):

```bash
kubectl apply -k deploy/k8s/overlays/dev
kubectl apply -k deploy/k8s/overlays/prod
```

GCP (Cloud Build -> Artifact Registry -> GKE):

```bash
gcloud builds submit --config deploy/gcp/cloudbuild.yaml \
  --substitutions=_ENV=prod,_CLUSTER=mldm-prod,_LOCATION=us-central1
```

The prod overlay layers in workload identity, the Cloud SQL Auth Proxy
sidecar, and External Secrets Operator wiring against Secret Manager.

## Tests

```bash
make test          # pytest (chat module unit tests + chat_graph integration)
make lint          # ruff check
make fmt           # ruff format
make type          # mypy
```

The DSPy-backed tests use a tiny in-process `dspy.LM` subclass / monkeypatched
modules — no API keys or network required.
