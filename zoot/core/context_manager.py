"""Prompt 组装与上下文预算控制。

这个模块负责决定：每一轮到底把多少 prefix、memory、相关笔记、历史
以及当前用户请求送进模型。
历史压缩主要发生在 turn_history（折叠旧工具输出）和 compact（跨轮次摘要）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..features import memory as memorylib, skills as skillslib
from .context_usage import ContextUsageAnalyzer
from .turn_history import TurnHistoryBuilder, tail_clip

DEFAULT_TOTAL_BUDGET = 60000
DEFAULT_SECTION_BUDGETS = {
    "prefix": 12000,
    "memory": 8000,
    "skills": 4000,
    "relevant_memory": 6000,
    "history": 30000,
}
SECTION_ORDER = ("prefix", "memory", "skills", "relevant_memory", "history", "current_request")
CURRENT_REQUEST_SECTION = "current_request"
RELEVANT_MEMORY_LIMIT = 3


@dataclass
class SectionRender:
    raw: str
    budget: int
    rendered: str
    details: dict | None = None

    @property
    def raw_chars(self):
        return len(self.raw)

    @property
    def rendered_chars(self):
        return len(self.rendered)


class ContextManager:
    def __init__(
        self,
        agent,
        total_budget=DEFAULT_TOTAL_BUDGET,
        section_budgets=None,
    ):
        self.agent = agent
        self.total_budget = int(total_budget)
        self.section_budgets = dict(DEFAULT_SECTION_BUDGETS)
        if section_budgets:
            self.section_budgets.update({str(key): int(value) for key, value in section_budgets.items()})
        self.history_builder = TurnHistoryBuilder(agent)

    def build(self, user_message):
        """按预算组装一轮完整 prompt。

        历史压缩策略已改为以 history 为主要压缩对象：
        - TurnHistoryBuilder 做渐进式工具输出折叠
        - CompactManager.compact() 做跨轮次摘要
        - read_file 超大输出在工具层做硬截断
        不再逐 section 裁剪预算。
        """
        user_message = str(user_message)
        memory_enabled = True
        relevant_memory_enabled = True
        if hasattr(self.agent, "feature_enabled"):
            memory_enabled = self.agent.feature_enabled("memory")
            relevant_memory_enabled = self.agent.feature_enabled("relevant_memory")
        memory_text = "Memory:\n- disabled" if not memory_enabled else str(self.agent.memory_text())
        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "memory": memory_text,
            "skills": skillslib.render_prompt_section(getattr(self.agent, "skills", {})),
            "history": "",
            CURRENT_REQUEST_SECTION: f"Current user request:\n{user_message}",
        }
        if hasattr(self.agent, "todo_ledger"):
            section_texts["memory"] += "\n\n" + self.agent.todo_ledger.render_prompt()
        checkpoint_text = ""
        if hasattr(self.agent, "render_checkpoint_text"):
            checkpoint_text = str(self.agent.render_checkpoint_text() or "").strip()
        if checkpoint_text:
            section_texts["memory"] += "\n\n" + checkpoint_text
        if memory_enabled and hasattr(self.agent, "memory_dir"):
            section_texts["memory"] += "\n\n" + memorylib.build_memory_system_section(self.agent.memory_dir)
        selected_notes = []
        if memory_enabled and relevant_memory_enabled and hasattr(self.agent, "memory") and hasattr(self.agent.memory, "retrieval_candidates"):
            selected_notes = self.agent.memory.retrieval_candidates(user_message, limit=RELEVANT_MEMORY_LIMIT)

        budgets = dict(self.section_budgets)
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
        prompt = self._assemble_prompt(rendered)

        metadata = self._metadata(
            prompt=prompt,
            rendered=rendered,
            budgets=budgets,
            selected_notes=selected_notes,
            user_message=user_message,
            section_texts=section_texts,
        )
        return prompt, metadata

    def _render_sections(self, section_texts, budgets, selected_notes=None):
        rendered = {}
        for section in SECTION_ORDER:
            budget = budgets.get(section)
            if section == CURRENT_REQUEST_SECTION:
                raw = section_texts[section]
                rendered[section] = SectionRender(raw=raw, budget=0, rendered=raw, details={})
            elif section == "relevant_memory":
                rendered[section] = self._render_relevant_memory(selected_notes or [], int(budget or 0))
            elif section == "history":
                rendered[section] = self._render_history_section(int(budget or 0))
            else:
                raw = section_texts[section]
                rendered_text = tail_clip(raw, int(budget)) if budget is not None else raw
                rendered[section] = SectionRender(raw=raw, budget=int(budget) if budget is not None else 0, rendered=rendered_text, details={})
        return rendered

    def _render_relevant_memory(self, selected_notes, budget):
        header = "Relevant process memory:"
        note_texts = []
        for note in selected_notes:
            text = str(note.get("text", "")).strip()
            if not text:
                continue
            note_type = note.get("type", "")
            prefix = f"{note_type}: " if note_type else ""
            note_texts.append(f"{prefix}{text}")
        raw_lines = [header] + [f"- {text}" for text in note_texts]
        raw = "\n".join(raw_lines) if note_texts else "\n".join([header, "- none"])
        if not note_texts:
            rendered = raw
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "selected_notes": [],
                    "rendered_notes": [],
                    "selected_count": 0,
                    "rendered_count": 0,
                    "note_budget": 0,
                },
            )

        per_note_budget = self._per_note_budget(budget, len(note_texts), header)
        rendered_notes = []
        while True:
            rendered_notes = [tail_clip(text, per_note_budget) for text in note_texts]
            rendered = "\n".join([header] + [f"- {text}" for text in rendered_notes])
            if len(rendered) <= budget or per_note_budget <= 1:
                break
            per_note_budget -= 1

        if len(rendered) > budget and budget > 0:
            rendered = tail_clip(raw, budget)
            rendered_notes = [rendered]

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "selected_notes": note_texts,
                "rendered_notes": rendered_notes,
                "selected_count": len(note_texts),
                "rendered_count": len(rendered_notes),
                "note_budget": per_note_budget,
            },
        )

    def _per_note_budget(self, budget, note_count, header):
        if note_count <= 0:
            return 0
        overhead = len(header) + 3 * note_count
        usable = max(0, budget - overhead)
        return max(1, usable // note_count)

    def _render_history_section(self, budget):
        history = list(getattr(self.agent, "session", {}).get("history", []))
        raw = self.history_builder.raw_text(history)
        if not history:
            rendered = "Transcript:\n- empty"
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "rendered_entries": [],
                    "older_entries_count": 0,
                    "collapsed_duplicate_reads": 0,
                    "reused_file_summary_count": 0,
                    "summarized_tool_count": 0,
                    "folded_tool_count": 0,
                    "rescue_count": 0,
                    "rendered_turns": 0,
                },
            )

        rendered, history_details = self.history_builder.render_section(
            budget, total_budget=self.total_budget
        )

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details=history_details,
        )

    def _assemble_prompt(self, rendered):
        return "\n\n".join(rendered[section].rendered for section in SECTION_ORDER).strip()

    def _metadata(self, prompt, rendered, budgets, selected_notes, user_message, section_texts):
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_chars": rendered[section].raw_chars,
                "budget_chars": int(budgets.get(section, 0)),
                "rendered_chars": rendered[section].rendered_chars,
            }
        section_metadata[CURRENT_REQUEST_SECTION] = {
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_chars": None,
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
        }
        return {
            "prompt_chars": len(prompt),
            "prompt_budget_chars": self.total_budget,
            "prompt_over_budget": len(prompt) > self.total_budget,
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                section: (None if section == CURRENT_REQUEST_SECTION else int(budgets.get(section, 0)))
                for section in SECTION_ORDER
            },
            "sections": section_metadata,
            "relevant_memory": {
                "limit": RELEVANT_MEMORY_LIMIT,
                "selected_count": len(selected_notes),
                "selected_notes": [note["text"] for note in selected_notes],
                "selected_sources": [str(note.get("source", "")).strip() for note in selected_notes],
                "selected_kinds": [str(note.get("kind", "episodic")).strip() or "episodic" for note in selected_notes],
                "selected_durable_count": sum(
                    1 for note in selected_notes if (str(note.get("kind", "episodic")).strip() or "episodic") == "durable"
                ),
                "raw_chars": rendered["relevant_memory"].raw_chars,
                "rendered_chars": rendered["relevant_memory"].rendered_chars,
                "rendered_notes": list(rendered["relevant_memory"].details.get("rendered_notes", [])),
                "rendered_count": int(rendered["relevant_memory"].details.get("rendered_count", 0)),
            },
            "history": {
                "raw_chars": rendered["history"].raw_chars,
                "rendered_chars": rendered["history"].rendered_chars,
                "older_entries_count": int(rendered["history"].details.get("older_entries_count", 0)),
                "collapsed_duplicate_reads": int(rendered["history"].details.get("collapsed_duplicate_reads", 0)),
                "reused_file_summary_count": int(rendered["history"].details.get("reused_file_summary_count", 0)),
                "summarized_tool_count": int(rendered["history"].details.get("summarized_tool_count", 0)),
                "rendered_turns": int(rendered["history"].details.get("rendered_turns", 0)),
            },
            "skills": self._skills_metadata(),
            "current_request": {
                "text": user_message,
                "raw_chars": len(user_message),
                "rendered_chars": len(user_message),
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
            },
            "context_usage": ContextUsageAnalyzer(self.agent).analyze(rendered),
        }

    def _skills_metadata(self):
        skills = getattr(self.agent, "skills", {})
        items = [skill.metadata() for skill in skillslib.list_skills(skills, user_invocable_only=False)]
        return {
            "available_count": len(items),
            "user_invocable_count": sum(1 for item in items if item["user_invocable"]),
            "items": items,
        }
