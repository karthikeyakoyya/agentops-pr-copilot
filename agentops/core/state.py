"""State schema shared by every node in the PR Copilot graph."""

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RunStatus:
    RUNNING = "running"
    BLOCKED = "blocked"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETE = "complete"
    DENIED = "denied"


class PRCopilotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_role: str
    repo_path: str
    proposed_fix: Optional[str]
    guardrail_passed: Optional[bool]
    status: str
