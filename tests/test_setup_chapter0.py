"""Editor settings written by the chapter 0 installer (VS Code / Cursor)."""

import json

from scripts.setup_chapter0 import has_editor_settings, write_editor_settings


def test_writes_portable_settings_for_fresh_checkout(tmp_path):
    assert write_editor_settings(tmp_path) is True
    settings = json.loads((tmp_path / ".vscode/settings.json").read_text())
    # Relative to the workspace, so moving or renaming the checkout keeps working.
    assert settings["python.defaultInterpreterPath"] == "${workspaceFolder}/.venv/bin/python"
    assert settings["python.analysis.extraPaths"] == [
        "${workspaceFolder}/chapter0_fundamentals/exercises"
    ]
    assert str(tmp_path) not in json.dumps(settings)
    # A rerun recognises its own file and doesn't nag the user to add settings.
    assert write_editor_settings(tmp_path) is False
    assert has_editor_settings(tmp_path) is True


def test_leaves_existing_settings_untouched(tmp_path):
    existing = tmp_path / ".vscode/settings.json"
    existing.parent.mkdir()
    existing.write_text('{\n  // my own JSONC settings\n  "editor.fontSize": 14\n}\n')
    assert write_editor_settings(tmp_path) is False
    assert "my own JSONC settings" in existing.read_text()
    assert has_editor_settings(tmp_path) is False  # JSONC: user is shown what to add
