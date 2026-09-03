"""Governance primitives: which roles can do what, and an audit trail
of every decision. Deliberately small — this shows the *shape* of
guardrails (deny-by-default on a role check, every action logged) rather
than being a complete policy engine.
"""

import json
import os
import time

AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl")

# Actions that mutate something (here: touching the repo) require a role
# beyond the default "viewer". Extend this map as new mutating actions
# are added — nothing else needs to change.
RESTRICTED_ACTIONS: dict[str, tuple[str, ...]] = {
    "apply_fix": ("maintainer",),
}


def is_allowed(action: str, user_role: str) -> bool:
    """True if `user_role` may perform `action`. Unlisted actions are
    unrestricted (available to any role, including the default 'viewer')."""
    allowed_roles = RESTRICTED_ACTIONS.get(action)
    return allowed_roles is None or user_role in allowed_roles


def audit(event: str, **fields) -> None:
    """Append a structured audit record. Never raises on I/O failure —
    an audit-log outage should never take down the agent."""
    record = {"ts": time.time(), "event": event, **fields}
    try:
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass
