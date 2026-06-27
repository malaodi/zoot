from pathlib import Path


def test_core_modules_stay_below_entropy_budget():
    root = Path(__file__).resolve().parents[1]
    budgets = {
        "zoot/core/runtime.py": 950,
        "zoot/core/runtime_events.py": 90,
        "zoot/core/runtime_consumers.py": 90,
        "zoot/core/artifacts.py": 130,
        "zoot/core/task_state.py": 140,
        "zoot/core/todo_ledger.py": 120,
        "zoot/core/worker_manager.py": 220,
        "zoot/core/context_manager.py": 420,
        "zoot/core/context_usage.py": 120,
        "zoot/core/compact.py": 180,
        "zoot/core/engine.py": 470,
        "zoot/core/model_errors.py": 100,
        "zoot/core/permissions.py": 140,
        "zoot/core/tool_policy.py": 90,
        "zoot/core/plan_mode.py": 140,
        "zoot/core/tool_executor.py": 181,
        "zoot/core/tool_profiles.py": 80,
        "zoot/core/turn_history.py": 250,
        "zoot/features/skills.py": 220,
        "zoot/features/skills_bundled.py": 120,
        "zoot/features/skills_runtime.py": 140,
        "zoot/tools/registry.py": 360,
        "zoot/tools/todos.py": 80,
        "zoot/tools/agents.py": 90,
    }

    for relative_path, max_lines in budgets.items():
        line_count = len((root / relative_path).read_text(encoding="utf-8").splitlines())
        assert line_count <= max_lines, f"{relative_path} has {line_count} lines, budget is {max_lines}"
