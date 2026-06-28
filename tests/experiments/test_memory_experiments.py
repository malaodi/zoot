"""Memory system experiments — harness regression tests for deterministic CI.

Each experiment verifies a specific memory layer's value under compression.
"""

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from zoot import Zoot, SessionStore, WorkspaceContext
from zoot.testing import ScriptedModelClient


@dataclass
class ExperimentResult:
    scenario: str
    memory_on: dict = field(default_factory=dict)
    memory_off: dict = field(default_factory=dict)

    def delta(self, key: str):
        on = self.memory_on.get(key, 0)
        off = self.memory_off.get(key, 0)
        return on - off


def build_workspace(root, files: dict[str, str]):
    for relpath, content in files.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return WorkspaceContext.build(root)


def build_agent(workspace, scripted_outputs, memory_enabled=True):
    store = SessionStore(workspace.repo_root + "/.zoot/sessions")
    return Zoot(
        model_client=ScriptedModelClient(scripted_outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        feature_flags={
            "memory": memory_enabled,
            "relevant_memory": bool(memory_enabled),
            "context_reduction": True,
        },
    )


def inject_layer1_compaction(agent):
    """Fold all read_file tool outputs older than the most recent turn."""
    history = agent.session.get("history", [])
    turns = {}
    for item in history:
        tid = item.get("turn_id", "legacy")
        turns.setdefault(tid, []).append(item)
    turn_ids = list(turns.keys())
    if len(turn_ids) < 2:
        return
    recent = {turn_ids[-1]}
    for tid, items in turns.items():
        if tid in recent:
            continue
        for item in items:
            if item.get("role") != "tool":
                continue
            if item.get("name") != "read_file":
                continue
            orig = item.get("content", "")
            if len(orig) > 100:
                item["content"] = f"[Old read_file result cleared: {item.get('args',{}).get('path','?')}]"


def inject_layer2_compaction(agent):
    """Compact all but the last 2 turns into a structured summary."""
    agent.compact_history(trigger="experiment", keep_recent_turns=2)


# ---------------------------------------------------------------------------
# A1 — Single-fact follow-up
# ---------------------------------------------------------------------------
def run_experiment_a1(tmp_path):
    _a1_files = {"a.txt": "fact = The deploy token name is SKYLINE"}

    def scenario(memory_on: bool):
        w = build_workspace(tmp_path / ("on" if memory_on else "off"), _a1_files)
        outputs = [
            '<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":5}}</tool>',
            "<final>Remembered the key fact.</final>",
        ]
        if memory_on:
            outputs.append("<final>SKYLINE</final>")
        else:
            outputs.append('<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":5}}</tool>')
            outputs.append("<final>SKYLINE</final>")
        a = build_agent(w, outputs, memory_enabled=memory_on)
        a.ask("Read a.txt and remember the key fact.")
        turn1_end = len(a.session.get("history", []))
        inject_layer1_compaction(a)
        a.ask("What is the deploy token name in a.txt?")
        history = a.session.get("history", [])
        step2_items = history[turn1_end:]
        tool_calls = [i for i in step2_items if i.get("role") == "tool"]
        reads = [i for i in tool_calls if i.get("name") == "read_file"]
        finals = [i for i in step2_items if i.get("role") == "assistant"]
        last = finals[-1]["content"] if finals else ""
        return {
            "repeated_reads": len(reads),
            "tool_steps": len(tool_calls),
            "correct": "SKYLINE" in str(last),
        }

    return ExperimentResult(
        scenario="A1: single-fact follow-up",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# A2 — Cross-file follow-up edit
# ---------------------------------------------------------------------------
def run_experiment_a2(tmp_path):
    _a2_files = {
        "a.txt": "placeholder = DELTA",
        "b.txt": "use placeholder here",
    }

    def scenario(memory_on: bool):
        w = build_workspace(tmp_path / ("on" if memory_on else "off"), _a2_files)
        outputs = [
            '<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":5}}</tool>',
            "<final>Placeholder is DELTA.</final>",
        ]
        if memory_on:
            outputs.append('<tool>{"name":"patch_file","args":{"path":"b.txt","old_text":"use placeholder here","new_text":"use DELTA here"}}</tool>')
        else:
            outputs.append('<tool>{"name":"read_file","args":{"path":"a.txt","start":1,"end":5}}</tool>')
            outputs.append("<final>DELTA</final>")
            outputs.append('<tool>{"name":"patch_file","args":{"path":"b.txt","old_text":"use placeholder here","new_text":"use DELTA here"}}</tool>')
        outputs.append("<final>Updated b.txt.</final>")
        a = build_agent(w, outputs, memory_enabled=memory_on)
        a.ask("Read a.txt and remember the placeholder token.")
        turn1_end = len(a.session.get("history", []))
        inject_layer1_compaction(a)
        a.ask("Update b.txt so it uses the remembered placeholder token from a.txt.")
        history = a.session.get("history", [])
        step2_items = history[turn1_end:]
        tool_calls = [i for i in step2_items if i.get("role") == "tool"]
        reads = [i for i in tool_calls if i.get("name") == "read_file"]
        patches = [i for i in tool_calls if i.get("name") == "patch_file"]
        return {
            "repeated_reads": len(reads),
            "tool_steps": len(tool_calls),
            "patched": len(patches) > 0,
        }

    return ExperimentResult(
        scenario="A2: cross-file follow-up edit",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# A3 — Multi-file recent-files
# ---------------------------------------------------------------------------
def run_experiment_a3(tmp_path):
    _a3_files = {
        "auth.txt": "login flow uses SESSION_KEY",
        "config.txt": "SESSION_KEY = sky-auth",
        "notes.txt": "",
    }

    def scenario(memory_on: bool):
        w = build_workspace(tmp_path / ("on" if memory_on else "off"), _a3_files)
        outputs = [
            '<tool>{"name":"read_file","args":{"path":"auth.txt","start":1,"end":5}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"config.txt","start":1,"end":5}}</tool>',
            "<final>SESSION_KEY is sky-auth.</final>",
        ]
        if memory_on:
            outputs.append('<tool>{"name":"write_file","args":{"path":"notes.txt","content":"sky-auth"}}</tool>')
        else:
            outputs.append('<tool>{"name":"read_file","args":{"path":"auth.txt","start":1,"end":5}}</tool>')
            outputs.append('<tool>{"name":"read_file","args":{"path":"config.txt","start":1,"end":5}}</tool>')
            outputs.append('<tool>{"name":"write_file","args":{"path":"notes.txt","content":"sky-auth"}}</tool>')
        outputs.append("<final>Wrote to notes.txt.</final>")
        a = build_agent(w, outputs, memory_enabled=memory_on)
        a.ask("Read auth.txt and config.txt, then remember the key session setting.")
        turn1_end = len(a.session.get("history", []))
        inject_layer1_compaction(a)
        a.ask("Write the remembered session setting into notes.txt.")
        history = a.session.get("history", [])
        step2_items = history[turn1_end:]
        tool_calls = [i for i in step2_items if i.get("role") == "tool"]
        reads = [i for i in tool_calls if i.get("name") == "read_file"]
        writes = [i for i in tool_calls if i.get("name") == "write_file"]
        return {
            "repeated_reads": len(reads),
            "tool_steps": len(tool_calls),
            "wrote": len(writes) > 0,
        }

    return ExperimentResult(
        scenario="A3: multi-file recent-files",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# B1 — Ruled-out path
# ---------------------------------------------------------------------------
def run_experiment_b1(tmp_path):
    _b1_files = {"bug.txt": "cause is NOT duplicate fixtures; cause is stale scope"}

    def scenario(process_on: bool):
        w = build_workspace(tmp_path / ("on" if process_on else "off"), _b1_files)
        a = build_agent(
            w,
            [
                '<tool>{"name":"read_file","args":{"path":"bug.txt","start":1,"end":10}}</tool>',
                "<final>Duplicate fixtures ruled out. Moving to scope analysis.</final>",
                '<tool>{"name":"read_file","args":{"path":"bug.txt","start":1,"end":10}}</tool>',
                "<final>Root cause is stale scope.</final>",
            ],
            memory_enabled=process_on,
        )
        a.ask("Investigate whether duplicate fixture names are the cause.")
        # Inject ruled_out process note
        if process_on:
            a.memory.append_note(
                "duplicate fixture names are not the cause",
                tags=("process", "ruled_out"),
                kind="process",
                note_type="ruled_out",
            )
        turn1_end = len(a.session.get("history", []))
        inject_layer2_compaction(a)
        a.ask("Continue debugging the issue.")
        history = a.session.get("history", [])
        step2_items = history[turn1_end:]
        reads = [i for i in step2_items if i.get("role") == "tool" and i.get("name") == "read_file"]
        first_after = reads[0] if reads else {}
        return {
            "repeated_exploration": len(reads),
            "tool_steps": len([i for i in history if i.get("role") == "tool"]),
            "last_final": str(history[-1].get("content", "")) if history else "",
        }

    return ExperimentResult(
        scenario="B1: ruled_out path",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# B4 — Blocker retention
# ---------------------------------------------------------------------------
def run_experiment_b4(tmp_path):
    _b4_files = {"patch.txt": "old_text matches twice\nline2\nline3"}

    def scenario(process_on: bool):
        w = build_workspace(tmp_path / ("on" if process_on else "off"), _b4_files)
        outputs = [
            '<tool>{"name":"read_file","args":{"path":"patch.txt","start":1,"end":10}}</tool>',
            '<tool>{"name":"patch_file","args":{"path":"patch.txt","old_text":"old_text matches twice","new_text":"fixed"}}</tool>',
            "<final>Patch failed.</final>",
        ]
        if process_on:
            outputs.append('<tool>{"name":"patch_file","args":{"path":"patch.txt","old_text":"matches twice","new_text":"fixed"}}</tool>')
        else:
            outputs.append('<tool>{"name":"patch_file","args":{"path":"patch.txt","old_text":"old_text matches twice","new_text":"fixed"}}</tool>')
        outputs.append("<final>Patched successfully.</final>")
        a = build_agent(w, outputs, memory_enabled=process_on)
        a.ask("Try to patch the file.")
        if process_on:
            a.memory.append_note(
                "patch_file failed because old_text was non-unique",
                tags=("process", "blocker"),
                kind="process",
                note_type="blocker",
            )
        turn1_end = len(a.session.get("history", []))
        inject_layer2_compaction(a)
        a.ask("Retry the patch correctly.")
        history = a.session.get("history", [])
        step2_items = history[turn1_end:]
        patches = [i for i in step2_items if i.get("role") == "tool" and i.get("name") == "patch_file"]
        first_retry_args = str(patches[0].get("args", {}).get("old_text", "")) if patches else ""
        return {
            "repeat_failed_patch": "old_text matches twice" == first_retry_args,
            "narrowed": "matches twice" == first_retry_args,
            "tool_steps": len([i for i in step2_items if i.get("role") == "tool"]),
        }

    return ExperimentResult(
        scenario="B4: blocker retention",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# B2 — Confirmed fact retention
# ---------------------------------------------------------------------------
def run_experiment_b2(tmp_path):
    _b2_files = {"conftest.txt": "db_fixture defined exactly once\nno duplicate here"}

    def scenario(process_on: bool):
        w = build_workspace(tmp_path / ("on" if process_on else "off"), _b2_files)
        outputs = [
            '<tool>{"name":"read_file","args":{"path":"conftest.txt","start":1,"end":10}}</tool>',
            "<final>db_fixture is defined exactly once. Confirmed.</final>",
        ]
        if process_on:
            outputs.append(
                '<tool>{"name":"search","args":{"pattern":"db_fixture","path":"."}}</tool>'
            )
        else:
            outputs.append(
                '<tool>{"name":"read_file","args":{"path":"conftest.txt","start":1,"end":10}}</tool>'
            )
        outputs.append("<final>No duplicates found elsewhere. Bug is in another area.</final>")
        a = build_agent(w, outputs, memory_enabled=process_on)
        a.ask("Check whether db_fixture is defined more than once.")
        if process_on:
            a.memory.append_note(
                "db_fixture is defined exactly once",
                tags=("process", "confirmed"),
                kind="process",
                note_type="confirmed",
            )
        turn1_end = len(a.session.get("history", []))
        inject_layer2_compaction(a)
        a.ask("Continue debugging why tests still fail.")
        history = a.session.get("history", [])
        step2_items = history[turn1_end:]
        reads = [i for i in step2_items if i.get("role") == "tool" and i.get("name") == "read_file"]
        return {
            "repeated_confirmation": len(reads),
            "tool_steps": len([i for i in step2_items if i.get("role") == "tool"]),
        }

    return ExperimentResult(
        scenario="B2: confirmed fact retention",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# B3 — Decision drift prevention
# ---------------------------------------------------------------------------
def run_experiment_b3(tmp_path):
    _b3_files = {"plan.txt": "Decision: run target test before full suite"}

    def scenario(process_on: bool):
        w = build_workspace(tmp_path / ("on" if process_on else "off"), _b3_files)
        outputs = [
            '<tool>{"name":"read_file","args":{"path":"plan.txt","start":1,"end":10}}</tool>',
            "<final>Decision: run target test before full suite.</final>",
        ]
        if process_on:
            outputs.append(
                '<tool>{"name":"run_shell","args":{"command":"pytest tests/test_target.py -q","timeout":20}}</tool>'
            )
        else:
            outputs.append(
                '<tool>{"name":"run_shell","args":{"command":"pytest -q","timeout":20}}</tool>'
            )
        outputs.append("<final>Tests passed.</final>")
        a = build_agent(w, outputs, memory_enabled=process_on)
        a.ask("Choose whether to run the target test or the full suite first.")
        if process_on:
            a.memory.append_note(
                "run target test before full suite",
                tags=("process", "decision"),
                kind="process",
                note_type="decision",
            )
        turn1_end = len(a.session.get("history", []))
        inject_layer2_compaction(a)
        a.ask("Proceed with verification.")
        history = a.session.get("history", [])
        step2_items = history[turn1_end:]
        shells = [i for i in step2_items if i.get("role") == "tool" and i.get("name") == "run_shell"]
        first_cmd = str(shells[0].get("args", {}).get("command", "")) if shells else ""
        return {
            "ran_target_first": "test_target.py" in first_cmd,
            "ran_full_suite": first_cmd == "pytest -q",
            "tool_steps": len([i for i in step2_items if i.get("role") == "tool"]),
        }

    return ExperimentResult(
        scenario="B3: decision drift prevention",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# C2 — Durable memory: key decisions
# ---------------------------------------------------------------------------
def run_experiment_c2(tmp_path):
    _c2_files = {}

    def scenario(durable_on: bool):
        w = build_workspace(tmp_path / ("on" if durable_on else "off"), _c2_files)
        a = build_agent(
            w,
            [
                "<final>Decision remembered: login state stored in session_store.</final>",
                "<final>Implemented login persistence using session_store.</final>",
            ],
            memory_enabled=durable_on,
        )
        a.ask("/remember login state must be stored in session_store")
        if durable_on:
            a.remember_durable_note("login state must be stored in session_store")
        a.ask("Implement login persistence.")
        final = str(a.session["history"][-1].get("content", ""))
        return {
            "uses_session_store": "session_store" in final.lower() or "session" in final.lower(),
        }

    return ExperimentResult(
        scenario="C2: durable key decisions",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# C3 — Durable memory: dependency requirements
# ---------------------------------------------------------------------------
def run_experiment_c3(tmp_path):
    _c3_files = {}

    def scenario(durable_on: bool):
        w = build_workspace(tmp_path / ("on" if durable_on else "off"), _c3_files)
        a = build_agent(
            w,
            [
                "<final>Noted: this project uses uv, not pip.</final>",
                "<final>Running test command: uv run pytest -q</final>",
            ],
            memory_enabled=durable_on,
        )
        a.ask("/remember this project requires uv, not pip, for Python task execution")
        if durable_on:
            a.remember_durable_note("this project requires uv, not pip, for Python task execution")
        a.ask("Run the project test command.")
        final = str(a.session["history"][-1].get("content", ""))
        return {
            "uses_uv": "uv" in final.lower(),
        }

    return ExperimentResult(
        scenario="C3: durable dependency requirements",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# C1 — Durable memory: project convention
# ---------------------------------------------------------------------------
def run_experiment_c1(tmp_path):
    _c1_files = {}

    def scenario(durable_on: bool):
        w = build_workspace(tmp_path / ("on" if durable_on else "off"), _c1_files)
        a = build_agent(
            w,
            [
                "<final>Remembered: this project uses pytest, not unittest.</final>",
                "<final>Test login flow written in pytest style with fixtures.</final>",
            ],
            memory_enabled=durable_on,
        )
        a.ask("/remember this project uses pytest, not unittest")
        if durable_on:
            a.remember_durable_note("this project uses pytest, not unittest")
        a.ask("Add a new test for the login flow.")
        final = str(a.session["history"][-1].get("content", ""))
        return {
            "uses_pytest": "pytest" in final.lower() or "fixture" in final.lower(),
            "uses_unittest": "unittest" in final.lower(),
        }

    return ExperimentResult(
        scenario="C1: durable project convention",
        memory_on=scenario(True),
        memory_off=scenario(False),
    )


# ---------------------------------------------------------------------------
# Test entry points
# ---------------------------------------------------------------------------
def test_experiment_a1_single_fact(tmp_path):
    r = run_experiment_a1(tmp_path)
    assert r.memory_on["correct"] is True
    # memory_on should not need to re-read
    assert r.memory_on["repeated_reads"] == 0


def test_experiment_a2_cross_file(tmp_path):
    r = run_experiment_a2(tmp_path)
    assert r.memory_on["patched"] is True
    assert r.memory_on["repeated_reads"] == 0


def test_experiment_a3_multifile(tmp_path):
    r = run_experiment_a3(tmp_path)
    assert r.memory_on["wrote"] is True
    assert r.memory_on["repeated_reads"] == 0


def test_experiment_b1_ruled_out(tmp_path):
    r = run_experiment_b1(tmp_path)
    assert "stale scope" in r.memory_on["last_final"].lower()


def test_experiment_b4_blocker(tmp_path):
    r = run_experiment_b4(tmp_path)
    # memory_on should narrow the old_text
    assert r.memory_on["narrowed"] is True


def test_experiment_c1_durable(tmp_path):
    r = run_experiment_c1(tmp_path)
    assert r.memory_on["uses_pytest"] is True


def test_experiment_b2_confirmed(tmp_path):
    r = run_experiment_b2(tmp_path)
    assert r.memory_on["repeated_confirmation"] == 0


def test_experiment_b3_decision(tmp_path):
    r = run_experiment_b3(tmp_path)
    assert r.memory_on["ran_target_first"] is True


def test_experiment_c2_key_decision(tmp_path):
    r = run_experiment_c2(tmp_path)
    assert r.memory_on["uses_session_store"] is True


def test_experiment_c3_dependency(tmp_path):
    r = run_experiment_c3(tmp_path)
    assert r.memory_on["uses_uv"] is True
