# AgentOps — PR Copilot

A platform-agnostic, governed LangGraph agent SDK, with a flagship
reference agent (**PR Copilot**) that reviews a pull request: retrieves
relevant code (RAG), runs the real test suite, proposes a fix, checks it
against a safety guardrail, and **pauses for human approval** before
it's allowed to touch the repo. Includes a live animated console
visualizing a run end-to-end.

## Honest status — read this before it goes anywhere near a resume

**Verified, for real, in this environment:**
- All 14 unit tests pass (`pytest tests/ -v`) — they test the pure
  governance/routing/guardrail logic and need no API key.
- The graph compiles with `interrupt_before=["apply_fix"]`, and the full
  human-in-the-loop lifecycle — pause, inspect `get_state().next`,
  resume with `invoke(None, config)` — was exercised end-to-end with a
  stub LLM standing in for a real model call. Also verified: a denied
  run never resumes into `apply_fix`, and a viewer-role run is still
  blocked by the RBAC check inside `apply_fix` even if something tried
  to resume it (defense in depth).
- The FastAPI app boots and its non-LLM endpoints (`/health`,
  `GET /runs/{id}` on a missing run) respond correctly.
- `console/index.html` was run in an actual headless browser
  (Playwright/Chromium), both demo scenarios completed with zero
  JS runtime errors, and the approve/deny interactions were exercised
  and screenshotted.

**Not verified / not real yet:**
- The agent's actual LLM call (`call_model` in `pr_copilot.py`) has
  never run against a live Anthropic or OpenAI model in this
  environment — no API key here. Wire your key via `.env` before
  trusting that path.
- `retrieve_context`'s Chroma index and the `sentence-transformers`
  embedding model haven't been installed/run here (heavy deps, skipped
  to keep the verification loop fast) — the code is written correctly
  against their documented APIs but hasn't executed.
- The console (`console/index.html`) is a **standalone demo** with two
  scripted example runs baked into its JS (`SCRIPTS.safe` /
  `SCRIPTS.risky`). It is not wired to the FastAPI backend. Treat it as
  "this is what the real thing would show," not "this is live."
- Nothing here is deployed anywhere.

Don't describe this as "deployed" or "in production" until you've
actually done that work — see "Next steps" below.

## How it maps to the Cognizant Python Gen AI Engineer JD

| JD requirement | Where it lives | Verified here? |
|---|---|---|
| Production-grade Python (not notebooks) | Whole `agentops/` package, `pyproject.toml` | Yes — installs, imports, tests pass |
| Agentic framework (LangGraph) | `agentops/agents/pr_copilot.py` — hand-built `StateGraph` | Yes — compiles, routes correctly |
| Tool/function calling | `agentops/tools/repo_tools.py`, bound via `.bind_tools()` | Structurally yes; live LLM call not run |
| RAG / embeddings | `retrieve_context` — Chroma index over the repo | Written, not executed (heavy deps) |
| Context/memory management | `MemorySaver` checkpointer, `thread_id`-keyed runs | Yes — resume-from-checkpoint verified |
| Platform-agnostic architecture | `agentops/core/llm_factory.py` — every other module calls `get_llm()`, never a vendor SDK directly | Yes, by construction |
| SDLC / CI-CD, and agents augmenting it | The whole product **is** an SDLC-augmentation agent; `.github/workflows/ci.yml` runs the real test suite on push | Yes — workflow file present, tests pass locally |
| API design / microservices / SDK | `agentops/api/main.py` (FastAPI), packaged as `agentops-sdk` via `pyproject.toml` | Yes — app boots, non-LLM routes verified |
| Agent evaluation / observability | `agentops/eval_harness.py` — uses `graph.stream(stream_mode="updates")`, exports traces | Written correctly against LangGraph's stream API; not run (needs LLM key) |
| Governance — guardrails, access control, auditability, human oversight | `core/governance.py` (RBAC + audit log), `_is_risky()` guardrail, `interrupt_before=["apply_fix"]` as the human-oversight gate | Yes — all four paths (safe/approved, safe/denied, risky/blocked, viewer/RBAC-denied) verified end-to-end |

## Project layout

```
agentops-pr-copilot/
├── agentops/
│   ├── core/
│   │   ├── llm_factory.py   # vendor-agnostic LLM selection
│   │   ├── state.py          # PRCopilotState, RunStatus
│   │   ├── governance.py     # RBAC + audit log
│   │   └── tracer.py          # per-node execution trace export
│   ├── tools/
│   │   └── repo_tools.py      # retrieve_context (RAG), run_tests
│   ├── agents/
│   │   └── pr_copilot.py      # the StateGraph
│   ├── api/
│   │   └── main.py             # FastAPI: /runs, /approve, /deny
│   └── eval_harness.py         # scenario-based eval + trace export
├── console/
│   └── index.html               # standalone animated demo console
├── tests/                        # 14 passing unit tests, no API key needed
├── .github/workflows/ci.yml     # runs pytest on push
├── pyproject.toml                # packaged as `agentops-sdk`
└── .env.example
```

## Running it

```bash
cp .env.example .env        # fill in ANTHROPIC_API_KEY or OPENAI_API_KEY
pip install -e ".[dev]"
pytest tests/ -v             # the 14 verified tests — no key needed
uvicorn agentops.api.main:app --reload
```

```bash
curl -X POST localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"repo_path": ".", "pr_diff": "checkout_flow_test times out on #submit-btn", "user_role": "maintainer"}'
# -> {"run_id": "...", "status": "awaiting_approval", "proposed_fix": "..."}

curl -X POST localhost:8000/runs/<run_id>/approve \
  -H "Content-Type: application/json" -d '{"actor": "you"}'
```

Open `console/index.html` directly in a browser (no build step, no
server) to see the animated demo — pick a scenario, hit "Run demo
review," and try Approve/Deny once it reaches the human-approval node.

## Next steps to make this genuinely resume-worthy

1. Wire a real `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and actually run
   `eval_harness.py` against a live model — confirm the agent's real
   tool-calling behavior, not just the routing logic around it.
2. Install the RAG dependencies (`langchain-chroma`,
   `langchain-huggingface`, `sentence-transformers`) and confirm
   `retrieve_context` actually indexes and retrieves from a real repo.
3. Wire `console/index.html` to the live FastAPI backend (swap the
   baked-in `SCRIPTS` object for `fetch()` calls to `/runs`) so the
   console shows real runs, not scripted ones.
4. Deploy the API somewhere (even a small Render/Fly.io instance) before
   calling this "production" anywhere.
