from langchain_core.messages import AIMessage
from langgraph.graph import END

from agentops.agents.pr_copilot import (
    apply_fix,
    guardrail_check,
    route_after_agent,
    route_after_guardrail,
)
from agentops.core.state import RunStatus


def _state(**overrides):
    base = {
        "messages": [AIMessage(content="Change the timeout from 5s to 15s in checkout_flow_test.")],
        "user_role": "viewer",
    }
    base.update(overrides)
    return base


def test_guardrail_passes_safe_fix():
    result = guardrail_check(_state())
    assert result["guardrail_passed"] is True
    assert result["status"] == RunStatus.AWAITING_APPROVAL


def test_guardrail_blocks_risky_fix():
    risky = _state(messages=[AIMessage(content="Just run DROP TABLE sessions; to clear it out.")])
    result = guardrail_check(risky)
    assert result["guardrail_passed"] is False
    assert result["status"] == RunStatus.BLOCKED


def test_guardrail_blocks_rm_rf():
    risky = _state(messages=[AIMessage(content="Simplest fix: rm -rf the build cache directory.")])
    result = guardrail_check(risky)
    assert result["guardrail_passed"] is False


def test_apply_fix_denies_viewer():
    state = _state(proposed_fix="bump timeout to 15s", user_role="viewer")
    result = apply_fix(state)
    assert result["status"] == RunStatus.BLOCKED
    assert "Denied" in result["messages"][0].content


def test_apply_fix_allows_maintainer():
    state = _state(proposed_fix="bump timeout to 15s", user_role="maintainer")
    result = apply_fix(state)
    assert result["status"] == RunStatus.COMPLETE
    assert "Applied fix" in result["messages"][0].content


def test_route_after_agent_goes_to_guardrail_without_tool_calls():
    ai_message = AIMessage(content="Here is the proposed fix.", tool_calls=[])
    assert route_after_agent({"messages": [ai_message]}) == "guardrail_check"


def test_route_after_agent_goes_to_tools_with_tool_calls():
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "run_tests", "args": {}, "id": "call_1"}],
    )
    assert route_after_agent({"messages": [ai_message]}) == "tools"


def test_route_after_guardrail_ends_when_blocked():
    assert route_after_guardrail({"guardrail_passed": False}) == END


def test_route_after_guardrail_proceeds_to_apply_fix_when_passed():
    assert route_after_guardrail({"guardrail_passed": True}) == "apply_fix"
