"""Every CLI command must at least parse its signature (catches broken annotations/imports)."""

import pytest
from typer.testing import CliRunner

from reeftrack.cli import app

COMMANDS = ["prepare", "add-right", "gui", "track", "replay", "reconstruct", "fuse", "view", "compare",
            "annotate", "library", "compress", "compress-batch", "sync-annotations", "check-install",
            "fetch-models"]


@pytest.mark.parametrize("cmd", COMMANDS)
def test_command_help(cmd):
    res = CliRunner().invoke(app, [cmd, "--help"])
    assert res.exit_code == 0, res.output


def test_check_install_reports_missing(monkeypatch):
    import importlib

    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda m, *a: (_ for _ in ()).throw(ImportError("No module named 'torch'")) if m == "torch"
                        else real(m, *a))
    res = CliRunner().invoke(app, ["check-install"])
    assert res.exit_code == 1 and "torch" in res.output
    assert CliRunner().invoke(app, ["check-install", "--no-tracking"]).exit_code == 0
