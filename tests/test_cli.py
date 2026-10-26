"""Every CLI command must at least parse its signature (catches broken annotations/imports)."""

import pytest
from typer.testing import CliRunner

from reeftrack.cli import app

COMMANDS = ["prepare", "add-right", "gui", "track", "replay", "reconstruct", "fuse", "view", "compare",
            "annotate", "library", "compress", "compress-batch"]


@pytest.mark.parametrize("cmd", COMMANDS)
def test_command_help(cmd):
    res = CliRunner().invoke(app, [cmd, "--help"])
    assert res.exit_code == 0, res.output
