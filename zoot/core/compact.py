"""Session compaction boundary."""

from .context_usage import estimate_tokens
from .workspace import now


class CompactManager:
    def __init__(self, agent):
        self.agent = agent

    def compact(self, trigger="manual", keep_recent_turns=2):
        history = list(self.agent.session.get("history", []))
        groups = self._group(history)
        if len(groups) <= keep_recent_turns:
            summary = self._summary(trigger, history, history, "")
            self.agent.session_event_bus.emit("compaction_created", summary)
            return summary

        compacted_turns = groups[:-keep_recent_turns]
        kept_turns = groups[-keep_recent_turns:]
        compacted_items = [item for _, items in compacted_turns for item in items]
        kept_items = [item for _, items in kept_turns for item in items]
        summary_text = self._summary_text(compacted_items)
        summary_item = self.agent.turn_history.enrich(
            {
                "role": "system",
                "kind": "compact_summary",
                "content": summary_text,
                "created_at": now(),
                "source": "compact",
            }
        )
        self.agent.session["history"] = [summary_item, *kept_items]
        summary = self._summary(trigger, history, self.agent.session["history"], summary_text)
        self.agent.session.setdefault("compactions", []).append(summary)
        self.agent.session_path = self.agent.session_store.save(self.agent.session)
        self.agent.session_event_bus.emit("compaction_created", summary)
        if self.agent.current_task_state:
            self.agent.emit_trace(self.agent.current_task_state, "compaction_started", {"trigger": trigger, "pre_tokens": summary["pre_tokens"]})
            self.agent.emit_trace(self.agent.current_task_state, "compaction_finished", summary)
        return summary

    @staticmethod
    def _group(history):
        groups = []
        by_id = {}
        for item in history:
            turn_id = str(item.get("turn_id") or "legacy")
            if turn_id not in by_id:
                by_id[turn_id] = []
                groups.append((turn_id, by_id[turn_id]))
            by_id[turn_id].append(item)
        return groups

    def _summary(self, trigger, before, after, summary_text):
        pre_chars = sum(len(str(item.get("content", ""))) for item in before)
        post_chars = sum(len(str(item.get("content", ""))) for item in after)
        return {
            "trigger": str(trigger),
            "created_at": now(),
            "pre_tokens": estimate_tokens(pre_chars),
            "post_tokens": estimate_tokens(post_chars),
            "pre_items": len(before),
            "post_items": len(after),
            "summary_chars": len(summary_text),
        }

    def _summary_text(self, items):
        memory = getattr(self.agent, "memory", None)
        memory_dict = memory.to_dict() if memory and hasattr(memory, "to_dict") else {}
        episodic = memory_dict.get("episodic_notes", [])
        working = memory_dict.get("working", {})
        task_summary = str(working.get("task_summary", ""))
        recent_files = working.get("recent_files", [])

        user_requests = []
        assistant_notes = []
        files_read = []
        files_modified = []
        for item in items:
            if item.get("role") == "user":
                user_requests.append(str(item.get("content", "")).strip())
            elif item.get("role") == "assistant":
                assistant_notes.append(str(item.get("content", "")).strip())
            elif item.get("role") == "tool":
                path = str(item.get("args", {}).get("path", "")).strip()
                if item.get("name") == "read_file" and path:
                    files_read.append(path)
                if item.get("name") in {"write_file", "patch_file"} and path:
                    files_modified.append(path)

        goal = task_summary or (user_requests[-1] if user_requests else "-")

        confirmed = [n["text"] for n in episodic if n.get("type") == "confirmed"]
        ruled_out = [n["text"] for n in episodic if n.get("type") == "ruled_out"]
        decisions = [n["text"] for n in episodic if n.get("type") == "decision"]
        blockers = [n["text"] for n in episodic if n.get("type") == "blocker"]

        constraints = ""
        for d in decisions:
            if d not in constraints:
                constraints += ("; " if constraints else "") + d
        for b in blockers:
            if b not in constraints:
                constraints += ("; " if constraints else "") + b

        checkpoint = getattr(self.agent, "render_checkpoint_text", None)
        next_step = "-"
        if checkpoint:
            ckpt_text = str(checkpoint() or "").strip()
            if ckpt_text:
                next_step = ckpt_text[:200]

        return "\n".join(
            [
                "Compacted task summary:",
                f"- Goal: {clip(goal, 200)}",
                f"- Constraints: {clip(constraints, 200) if constraints else '-'}",
                f"- Confirmed facts: {', '.join(confirmed) if confirmed else '-'}",
                f"- Ruled-out paths: {', '.join(ruled_out) if ruled_out else '-'}",
                f"- Decisions made: {', '.join(decisions) if decisions else '-'}",
                f"- Open blockers: {', '.join(blockers) if blockers else '-'}",
                f"- Key files: {', '.join(recent_files[:8]) if recent_files else ', '.join(sorted(set(files_read + files_modified))[:8]) or '-'}",
                f"- Next step: {clip(next_step, 200)}",
                f"- Compacted {len(items)} history items; preserved latest turns for exact wording",
            ]
        )


def clip(text, limit):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
