"""Memory system experiments — real provider validation.

These experiments use the actual configured model provider to verify
that memory layers provide real value under compression.

Set ZOOT_LIVE_MEMORY_EXPERIMENTS=1 to enable.
Requires a valid provider configuration (API key, model).
"""

import os
import textwrap
import sys
from pathlib import Path

import pytest

from zoot import Zoot, SessionStore, WorkspaceContext
from zoot.cli import _build_model_client

LIVE_FLAG = "ZOOT_LIVE_MEMORY_EXPERIMENTS"
SKIP_REASON = f"set {LIVE_FLAG}=1 to run real-provider memory experiments"


class FakeArgs:
    provider = os.environ.get("ZOOT_PROVIDER", "deepseek")
    model = os.environ.get("ZOOT_MODEL", None)
    base_url = os.environ.get("ZOOT_BASE_URL", None)
    api_key = os.environ.get("ZOOT_API_KEY", None)
    cwd = os.environ.get("ZOOT_CWD", ".")
    config = os.environ.get("ZOOT_CONFIG", None)
    temperature = float(os.environ.get("ZOOT_TEMPERATURE", "0.2"))
    openai_timeout = int(os.environ.get("ZOOT_TIMEOUT", "300"))


def build_workspace(root, files: dict[str, str]):
    for relpath, content in files.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return WorkspaceContext.build(root)


def build_live_agent(workspace, memory_enabled=True):
    try:
        client = _build_model_client(FakeArgs())
    except Exception as exc:
        pytest.skip(f"Cannot build model client: {exc}")

    store = SessionStore(workspace.repo_root + "/.zoot/sessions")
    return Zoot(
        model_client=client,
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        feature_flags={
            "memory": memory_enabled,
            "relevant_memory": bool(memory_enabled),
            "context_reduction": True,
        },
        max_steps=10,
    )


def _should_run():
    if os.environ.get(LIVE_FLAG) != "1":
        return False
    if not (os.environ.get("ZOOT_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        return False
    return True


# ---------------------------------------------------------------------------
# A1 Provider — Single-fact follow-up
# ---------------------------------------------------------------------------
def test_provider_a1_single_fact(tmp_path):
    if not _should_run():
        pytest.skip(SKIP_REASON)

    _files = {"a.txt": "fact = The deploy token name is SKYLINE"}
    w = build_workspace(tmp_path, _files)
    a = build_live_agent(w, memory_enabled=True)
    a.ask("Read a.txt and remember the key fact.")
    a.ask("What is the deploy token name in a.txt?")

    history = a.session.get("history", [])
    tool_calls = [i for i in history if i.get("role") == "tool"]
    reads = [i for i in tool_calls if i.get("name") == "read_file"]
    finals = [i for i in history if i.get("role") == "assistant"]
    last = str(finals[-1]["content"]) if finals else ""

    assert len(reads) <= 2, f"Expected at most 2 reads, got {len(reads)}"
    assert "SKYLINE" in last.upper()


# ---------------------------------------------------------------------------
# A2 Provider — Cross-file follow-up edit
# ---------------------------------------------------------------------------
def test_provider_a2_cross_file(tmp_path):
    if not _should_run():
        pytest.skip(SKIP_REASON)

    _files = {"a.txt": "placeholder = DELTA", "b.txt": "use placeholder here"}
    w = build_workspace(tmp_path, _files)
    a = build_live_agent(w, memory_enabled=True)
    a.ask("Read a.txt and remember the placeholder token.")
    a.ask("Update b.txt so it uses the remembered placeholder token from a.txt.")

    updated = (tmp_path / "b.txt").read_text(encoding="utf-8")
    assert "DELTA" in updated


# ---------------------------------------------------------------------------
# B4 Provider — Blocker retention
# ---------------------------------------------------------------------------
def test_provider_b4_blocker(tmp_path):
    if not _should_run():
        pytest.skip(SKIP_REASON)

    _files = {"patch.txt": "old_text matches twice\nline2\nline3"}
    w = build_workspace(tmp_path, _files)
    a = build_live_agent(w, memory_enabled=True)
    a.ask("Try to patch patch.txt by replacing 'old_text matches twice' with 'fixed'. Expect it to fail.")
    a.ask("Retry the patch correctly. Narrow the old_text if needed.")

    content = (tmp_path / "patch.txt").read_text(encoding="utf-8")
    assert "fixed" in content


# ---------------------------------------------------------------------------
# C1 Provider — Durable memory: project convention
# ---------------------------------------------------------------------------
def test_provider_c1_durable(tmp_path):
    if not _should_run():
        pytest.skip(SKIP_REASON)

    w = build_workspace(tmp_path, {})
    a = build_live_agent(w, memory_enabled=True)
    a.ask("/remember this project uses pytest, not unittest. Note this for later.")
    a.ask("Add a new test for the login flow. Write it to login_test.py")

    result = (tmp_path / "login_test.py").read_text(encoding="utf-8") if (tmp_path / "login_test.py").exists() else ""
    # Even if file not created, the model's answer should mention pytest
    history = a.session.get("history", [])
    finals = [i for i in history if i.get("role") == "assistant"]
    last = str(finals[-1]["content"]) if finals else ""
    assert ("pytest" in last.lower()) or ("pytest" in result.lower())
