"""Minimal execution tracer.

Feeds two things: eval_harness.py's pass/latency reporting, and the
AgentOps Console demo data (a trace export is exactly the shape the
console's timeline expects — see console/README notes). Swap this for
LangSmith by setting LANGCHAIN_TRACING_V2=true; nothing here needs to
change since LangSmith traces LangGraph runs automatically.
"""

import json
import os
import time
from typing import Any


class Tracer:
    def __init__(self, run_id: str, out_dir: str = "traces"):
        self.run_id = run_id
        self.out_dir = out_dir
        self.events: list[dict[str, Any]] = []

    def record(self, node: str, event_type: str, detail: Any = None) -> None:
        self.events.append(
            {
                "run_id": self.run_id,
                "node": node,
                "type": event_type,
                "detail": detail,
                "ts": time.time(),
            }
        )

    def export(self) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.events, f, indent=2)
        return path
