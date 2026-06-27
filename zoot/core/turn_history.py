"""Turn-aware transcript rendering."""

import json
import re
from collections import OrderedDict


def tail_clip(text, limit):
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


_FOLD_TOOL_OUTPUT_LENGTH_THRESHOLD = 200
_RETENTION_BUDGET_RATIO = 0.25
_MAX_RECENT_WINDOW = 6


def fold_tool_result(item, memory, workspace_root):
    """Progressive folding: replace old tool output body with a skeleton.

    Preserves the turn structure while removing the bulk of old tool output.
    Returns a short replacement string with enough context for the model to
    understand what happened without reading hundreds of lines.
    """
    name = str(item.get("name", "")).strip()
    args = item.get("args", {})
    content = str(item.get("content", ""))

    if name == "read_file":
        path = str(args.get("path", "")).strip()
        start = str(args.get("start", "1"))
        end = str(args.get("end", "?"))
        summary_info = ""
        if memory is not None and hasattr(memory, "to_dict"):
            sm = memory.to_dict().get("file_summaries", {}).get(path, {})
            if sm.get("summary"):
                summary_info = f" | summary: {sm['summary']}"
        return f"[Old read_file result cleared: {path} lines {start}-{end}{summary_info}]"

    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        search_path = str(args.get("path", ".")).strip()
        return f"[Old search result cleared: pattern=\"{pattern}\" in {search_path} | matches were previously inspected]"

    if name == "run_shell":
        command = str(args.get("command", "")).strip() or "shell"
        exit_info = ""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("exit_code:"):
                exit_info = f" | {line}"
                break
        ws_changed = False
        metadata = item.get("metadata", {})
        if metadata.get("workspace_changed"):
            ws_changed = True
        ws_info = " | workspace_changed=true" if ws_changed else ""
        return f"[Old run_shell result cleared: command=\"{command}\"{exit_info}{ws_info}]"

    return f"[Old {name} result cleared]"


def rescue_before_fold(item, memory):
    """Scan tool result for key signals before folding, preserve as process notes.

    If the tool output contains critical information (bug patterns, tracebacks,
    unique matches, etc.) that hasn't been recorded in process memory yet,
    write a process note before the output is folded away.
    """
    if memory is None or not hasattr(memory, "append_note"):
        return

    name = str(item.get("name", "")).strip()
    content = str(item.get("content", ""))
    args = item.get("args", {})

    if name == "run_shell":
        has_traceback = bool(re.search(r"Traceback \(most recent call last\)", content))
        has_error = "exit_code: 1" in content or "exit_code: 2" in content
        if has_traceback or has_error:
            command = str(args.get("command", "")).strip() or "unknown"
            error_lines = [line.strip() for line in content.splitlines()
                          if line.strip() and (has_traceback or "Error" in line)
                          and not line.startswith("exit_code")][:3]
            error_summary = " | ".join(error_lines) if error_lines else "non-zero exit"
            memory.append_note(
                f"run_shell error: {error_summary}",
                tags=("process", "blocker", command[:50]),
                source="folding_rescue",
                kind="process",
                note_type="blocker",
            )

    if name == "search":
        match_count = content.count("\n") + 1 if content and content != "(no matches)" else 0
        if match_count == 0:
            pattern = str(args.get("pattern", "")).strip()
            memory.append_note(
                f"search returned no matches for \"{pattern}\"",
                tags=("process", "ruled_out", pattern[:50]),
                source="folding_rescue",
                kind="process",
                note_type="ruled_out",
            )
        elif match_count == 1:
            memory.append_note(
                f"search found exactly one match: {content[:120]}",
                tags=("process", "confirmed"),
                source="folding_rescue",
                kind="process",
                note_type="confirmed",
            )

    if name == "read_file":
        has_exception = "Exception" in content or "Error" in content
        if has_exception:
            path = str(args.get("path", "")).strip()
            memory.append_note(
                f"read_file found error/exception in {path}",
                tags=("process", "blocker", path),
                source="folding_rescue",
                kind="process",
                note_type="blocker",
            )


class TurnHistoryBuilder:
    def __init__(self, agent):
        self.agent = agent

    def enrich(self, item):
        item = dict(item)
        if not item.get("turn_id"):
            current_turn = str(getattr(self.agent, "current_turn_id", "") or "")
            if not current_turn:
                if item.get("role") == "user" or not self.agent.session.get("_manual_turn_id"):
                    self.agent.session["_manual_turn_seq"] = int(self.agent.session.get("_manual_turn_seq", 0)) + 1
                    self.agent.session["_manual_turn_id"] = f"manual_{self.agent.session['_manual_turn_seq']:06d}"
                current_turn = str(self.agent.session.get("_manual_turn_id", "legacy"))
            item["turn_id"] = current_turn
        if not item.get("run_id"):
            item["run_id"] = str(getattr(self.agent, "current_run_id", "") or "")
        if not item.get("event_id"):
            self.agent.session["_event_seq"] = int(self.agent.session.get("_event_seq", 0)) + 1
            item["event_id"] = f"event_{self.agent.session['_event_seq']:06d}"
        item.setdefault("source", "runtime")
        return item

    def raw_text(self, history):
        if not history:
            return "Transcript:\n- empty"
        return "\n".join(["Transcript:", *self._render_turn_lines(history, line_limit=2000)])

    def render_section(self, budget, total_budget=None):
        history = list(getattr(self.agent, "session", {}).get("history", []))
        raw = self.raw_text(history)
        if not history:
            return raw, {
                "rendered_entries": [],
                "older_entries_count": 0,
                "collapsed_duplicate_reads": 0,
                "reused_file_summary_count": 0,
                "summarized_tool_count": 0,
                "folded_tool_count": 0,
                "rescue_count": 0,
                "rendered_turns": 0,
            }

        turns = self._group_turns(history)
        recent_window = self._compute_recent_window(turns, budget, total_budget)
        recent_turns = set(list(turns)[-recent_window:])
        entries, details = self._compressed_turn_entries(turns, recent_turns)
        details["recent_window"] = recent_window
        rendered_entries = []
        for entry in reversed(entries):
            candidate = entry["lines"] + rendered_entries
            if len("\n".join(["Transcript:", *candidate])) <= budget:
                rendered_entries = candidate
                continue
            if entry["turn_id"] in recent_turns:
                clipped = [tail_clip(line, max(40, budget // max(1, len(entry["lines"])))) for line in entry["lines"]]
                candidate = clipped + rendered_entries
                if len("\n".join(["Transcript:", *candidate])) <= budget:
                    rendered_entries = candidate
        rendered = "\n".join(["Transcript:", *rendered_entries])
        if len(rendered) > budget and budget > 0:
            rendered = tail_clip(raw, budget)
        details["rendered_entries"] = rendered_entries
        details["rendered_turns"] = sum(1 for line in rendered_entries if line.startswith("Turn "))
        return rendered, details

    def _compute_recent_window(self, turns, budget, total_budget):
        """Compute dynamic retention window: retain recent turns until their
        cumulative token count does not exceed ~25% of the total budget.
        Falls back to 3 if unable to compute. Capped at _MAX_RECENT_WINDOW.
        """
        if not total_budget or total_budget <= 0:
            return 3
        retention_budget = int(total_budget * _RETENTION_BUDGET_RATIO)
        if retention_budget <= 0:
            return 3
        turn_ids = list(turns)
        cumulative = 0
        window = 0
        for turn_id in reversed(turn_ids):
            items = turns[turn_id]
            turn_chars = sum(len(str(item.get("content", ""))) for item in items)
            cumulative += turn_chars
            window += 1
            if cumulative >= retention_budget:
                break
        return max(1, min(window, _MAX_RECENT_WINDOW))

    def _group_turns(self, history):
        turns = OrderedDict()
        for item in history:
            turn_id = str(item.get("turn_id") or "legacy")
            turns.setdefault(turn_id, []).append(item)
        return turns

    def _compressed_turn_entries(self, turns, recent_turns):
        entries = []
        seen_older_reads = set()
        details = {
            "recent_window": len(recent_turns),
            "older_entries_count": 0,
            "collapsed_duplicate_reads": 0,
            "reused_file_summary_count": 0,
            "summarized_tool_count": 0,
            "folded_tool_count": 0,
            "rescue_count": 0,
        }
        memory = getattr(self.agent, "memory", None)
        workspace_root = getattr(self.agent, "root", None)

        for turn_id, items in turns.items():
            recent = turn_id in recent_turns and any(item.get("role") != "tool" for item in items)
            lines = [f"Turn {turn_id}:"]
            for item in items:
                if item.get("kind") == "compact_summary":
                    lines.extend(str(item.get("content", "")).splitlines())
                    continue
                if not recent and item.get("role") == "tool" and item.get("name") == "read_file":
                    path = str(item.get("args", {}).get("path", "")).strip()
                    if path in seen_older_reads:
                        details["collapsed_duplicate_reads"] += 1
                        continue
                    seen_older_reads.add(path)
                    summary = self._reusable_file_summary(path)
                    if summary:
                        lines.append(f"{path} -> {summary}")
                        details["reused_file_summary_count"] += 1
                        continue
                if not recent and item.get("role") == "tool":
                    content_len = len(str(item.get("content", "")))
                    if content_len > _FOLD_TOOL_OUTPUT_LENGTH_THRESHOLD:
                        rescue_before_fold(item, memory)
                        details["rescue_count"] += 1
                        folded = fold_tool_result(item, memory, workspace_root)
                        lines.append(folded)
                        details["folded_tool_count"] += 1
                    else:
                        lines.append(self._summarize_old_tool_item(item))
                        details["summarized_tool_count"] += 1
                    continue
                lines.extend(self._render_item(item, 900 if recent else 80))
            if not recent:
                details["older_entries_count"] += 1
            entries.append({"turn_id": turn_id, "lines": lines})
        return entries, details

    def _render_turn_lines(self, history, line_limit):
        lines = []
        for turn_id, items in self._group_turns(history).items():
            lines.append(f"Turn {turn_id}:")
            for item in items:
                lines.extend(self._render_item(item, line_limit))
        return lines

    def _render_item(self, item, line_limit):
        if item.get("kind") == "compact_summary":
            return str(item.get("content", "")).splitlines()
        if item.get("role") == "tool":
            prefix = f"[tool:{item.get('name', '')}] {json.dumps(item.get('args', {}), sort_keys=True)}"
            content = tail_clip(item.get("content", ""), max(20, line_limit))
            return [prefix, content]
        return [f"[{item.get('role', '')}] {tail_clip(item.get('content', ''), line_limit)}"]

    def _reusable_file_summary(self, path):
        memory = getattr(self.agent, "memory", None)
        if memory is None or not hasattr(memory, "to_dict"):
            return ""
        summary = memory.to_dict().get("file_summaries", {}).get(str(path), {})
        return str(summary.get("summary", "")).strip()

    def _summarize_old_tool_item(self, item):
        if item.get("name") == "run_shell":
            command = str(item.get("args", {}).get("command", "")).strip() or "shell"
            lines = [line.strip() for line in str(item.get("content", "")).splitlines() if line.strip()]
            return f"{command} -> {' | '.join(lines[:3]) if lines else '(empty)'}"
        return self._render_item(item, 80)[0]
