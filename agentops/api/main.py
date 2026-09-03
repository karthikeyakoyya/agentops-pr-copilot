"""FastAPI service exposing PR Copilot as a small REST API.

Three endpoints carry the whole human-in-the-loop lifecycle:
  POST /runs                  start a review, runs until it needs a human
  POST /runs/{run_id}/approve  resume execution past the interrupt
  POST /runs/{run_id}/deny     record a denial; the graph is never resumed
"""

import uuid

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ..agents.pr_copilot import build_graph
from ..core.governance import audit

app = FastAPI(title="AgentOps — PR Copilot")
_graph = build_graph()
_denied_runs: dict[str, str] = {}


class StartRunRequest(BaseModel):
    repo_path: str
    pr_diff: str
    user_role: str = "viewer"


class DecisionRequest(BaseModel):
    reason: str = ""
    actor: str = "reviewer"


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _status_for(run_id: str) -> str:
    if run_id in _denied_runs:
        return "denied"
    snapshot = _graph.get_state(_config(run_id))
    if snapshot.next == ("apply_fix",):
        return "awaiting_approval"
    if not snapshot.next:
        return "blocked_or_complete"
    return "running"


@app.post("/runs")
def start_run(req: StartRunRequest):
    run_id = uuid.uuid4().hex[:12]
    config = _config(run_id)
    initial_state = {
        "messages": [HumanMessage(content=f"Review this PR diff and propose a fix if tests fail:\n{req.pr_diff}")],
        "user_role": req.user_role,
        "repo_path": req.repo_path,
    }
    result = _graph.invoke(initial_state, config=config)
    return {
        "run_id": run_id,
        "status": _status_for(run_id),
        "proposed_fix": result.get("proposed_fix"),
    }


@app.post("/runs/{run_id}/approve")
def approve_run(run_id: str, req: DecisionRequest):
    snapshot = _graph.get_state(_config(run_id))
    if snapshot.next != ("apply_fix",):
        raise HTTPException(400, "Run is not awaiting approval.")
    audit("human_approval_resolved", run_id=run_id, decision="approved", actor=req.actor)
    result = _graph.invoke(None, config=_config(run_id))
    return {"run_id": run_id, "status": _status_for(run_id), "result": result["messages"][-1].content}


@app.post("/runs/{run_id}/deny")
def deny_run(run_id: str, req: DecisionRequest):
    snapshot = _graph.get_state(_config(run_id))
    if snapshot.next != ("apply_fix",):
        raise HTTPException(400, "Run is not awaiting approval.")
    _denied_runs[run_id] = req.reason
    audit("human_approval_resolved", run_id=run_id, decision="denied", actor=req.actor, reason=req.reason)
    return {"run_id": run_id, "status": "denied", "reason": req.reason}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    snapshot = _graph.get_state(_config(run_id))
    if not snapshot.values:
        raise HTTPException(404, "Run not found.")
    return {
        "run_id": run_id,
        "status": _status_for(run_id),
        "proposed_fix": snapshot.values.get("proposed_fix"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
