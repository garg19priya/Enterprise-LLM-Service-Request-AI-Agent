"""What the agent knows during one run. Updated after every tool result."""
from dataclasses import dataclass, field


@dataclass
class AgentState:
    request_id: str
    request_text: str
    steps: int = 0
    tool_history: list = field(default_factory=list)   # {"name", "arguments", "result"}
    evidence_ids: list = field(default_factory=list)   # what the final answer may cite
    approvals: list = field(default_factory=list)      # {"action", "approved", "approver"}
    errors: list = field(default_factory=list)
    final_answer: dict = None
    stop_reason: str = None

    def record_tool(self, name: str, arguments: dict, result: dict) -> None:
        self.tool_history.append({"name": name, "arguments": arguments, "result": result})
        if not isinstance(result, dict) or "error" in result:
            if isinstance(result, dict):
                self.errors.append(result.get("error"))
            return
        if name == "read_status" and result.get("service"):
            self._add_evidence(f"status:{result['service']}")
        elif name == "search_knowledge":
            for hit in result.get("results", []):
                self._add_evidence(hit.get("note_id"))
        elif name == "record_proposed_action" and result.get("action_id"):
            self._add_evidence(result["action_id"])

    def _add_evidence(self, item):
        if item and item not in self.evidence_ids:
            self.evidence_ids.append(item)

    @property
    def tools_called(self) -> list:
        return [t["name"] for t in self.tool_history]

    def summary(self) -> dict:
        return {"request_id": self.request_id, "steps": self.steps, "tools_called": self.tools_called,
                "evidence_ids": self.evidence_ids, "errors": self.errors, "approvals": self.approvals,
                "stop_reason": self.stop_reason}
