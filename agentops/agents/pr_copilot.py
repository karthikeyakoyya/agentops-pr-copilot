"""LangGraph StateGraph for the PR Copilot agent.

Flow:
    agent <-> tools            (agent calls retrieve_context / run_tests
                                 as many times as it needs)
    agent -> guardrail_check    (once it responds without a tool call,
                                 its response is the proposed fix)
    guardrail_check -> END      (if the fix trips the risk guardrail)
    guardrail_check -> apply_fix (otherwise — but see below)

The graph is compiled with `interrupt_before=["apply_fix"]`: LangGraph
pauses execution right before that node runs and returns control to the
caller. That pause *is* the human-approval gate — no hand-rolled polling
loop, just the framework's own resume primitive. The API layer resumes
execution with `graph.invoke(None, config)` on approval, or simply never
resumes (and records a denial) if a human rejects it.
"""

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..core.governance import audit, is_allowed
from ..core.llm_factory import get_llm
from ..core.state import PRCopilotState, RunStatus
from ..tools.repo_tools import TOOLS

SYSTEM_PROMPT = """You are PR Copilot, an SDLC agent that reviews a pull \
request diff and proposes a fix for failing tests.

Rules:
- Call retrieve_context ONCE before proposing a fix, to ground it in the \
actual codebase rather than guessing.
- Call run_tests ONCE to see the current failure.
- Do not call either tool more than once each, and do not call a tool \
again just to double-check — you have a limited number of calls available.
- After calling each tool once, give your final proposed fix immediately.
- Never claim a fix is correct without evidence from the tools.
- Your final response (once you stop calling tools) is treated as the \
proposed fix — state clearly what changes, in which file, and why.
"""

_tools_by_name = {t.name: t for t in TOOLS}

# Keywords that trip the safety guardrail. A real system would use a
# proper static-analysis pass; this is intentionally simple to keep the
# control flow legible.
_RISKY_PATTERNS = ("drop table", "rm -rf", "delete from", "truncate table")


def _stringify_content(content) -> str:
    """AIMessage.content is normally a str, but some providers (Gemini
    via langchain_google_genai, notably) can return a list of content
    blocks instead. Normalize either shape to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _is_risky(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _RISKY_PATTERNS)


def call_model(state: PRCopilotState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    llm = get_llm().bind_tools(TOOLS)
    response = llm.invoke(messages)
    return {"messages": [response]}


def execute_tools(state: PRCopilotState):
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)

    results = []
    for call in last.tool_calls:
        name, args, call_id = call["name"], dict(call["args"]), call["id"]
        args.setdefault("repo_path", state.get("repo_path", "."))
        output = _tools_by_name[name].invoke(args)
        audit("tool_call", tool=name, args={k: v for k, v in args.items() if k != "repo_path"})
        results.append(ToolMessage(content=str(output), tool_call_id=call_id))
    return {"messages": results}


def route_after_agent(state: PRCopilotState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "guardrail_check"


def guardrail_check(state: PRCopilotState):
    proposed_fix = _stringify_content(state["messages"][-1].content)
    passed = not _is_risky(proposed_fix)
    audit("guardrail_check", passed=passed)
    return {
        "proposed_fix": proposed_fix,
        "guardrail_passed": passed,
        "status": RunStatus.AWAITING_APPROVAL if passed else RunStatus.BLOCKED,
    }


def route_after_guardrail(state: PRCopilotState) -> str:
    return "apply_fix" if state["guardrail_passed"] else END


def apply_fix(state: PRCopilotState):
    """Only reached once the graph has been resumed past the
    interrupt — i.e. a human has approved. RBAC is still checked here
    too, as defense in depth: never trust that "we got this far" alone
    means the caller is authorized."""
    role = state.get("user_role", "viewer")
    if not is_allowed("apply_fix", role):
        audit("permission_denied", tool="apply_fix", user_role=role)
        return {
            "status": RunStatus.BLOCKED,
            "messages": [AIMessage(content="Denied: apply_fix requires the 'maintainer' role.")],
        }
    audit("tool_call_authorized", tool="apply_fix", user_role=role)
    return {
        "status": RunStatus.COMPLETE,
        "messages": [AIMessage(content=f"Applied fix: {state['proposed_fix'][:200]}")],
    }


def build_graph():
    graph = StateGraph(PRCopilotState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_node("guardrail_check", guardrail_check)
    graph.add_node("apply_fix", apply_fix)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "guardrail_check": "guardrail_check"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("guardrail_check", route_after_guardrail, {"apply_fix": "apply_fix", END: END})
    graph.add_edge("apply_fix", END)

    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["apply_fix"])