"""Tests for the RoboAudit command-line interface."""

from pathlib import Path
import json
import pytest

from roboaudit.cli import main


class TestCommandLineInterface:
    """Test suite for CLI commands."""

    def test_cli_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "roboaudit" in captured.out
        assert "audit" in captured.out
        assert "batch" in captured.out

    def test_cli_audit_demo_episode(self, capsys):
        code = main(["audit", "sample_data/demo_episode"])
        assert code == 0
        captured = capsys.readouterr()
        assert "RoboAudit Quality Report: demo_episode" in captured.out
        assert "Quality Score:" in captured.out

    def test_cli_audit_with_json_and_validation(self, tmp_path):
        json_out = tmp_path / "report.json"
        code = main(["audit", "sample_data/demo_episode", "--output", str(json_out)])
        assert code == 0
        assert json_out.is_file()

        val_code = main(["validate", str(json_out)])
        assert val_code == 0

    def test_cli_visualize(self, tmp_path):
        html_out = tmp_path / "board.html"
        code = main(["visualize", "sample_data/demo_episode", "--output", str(html_out)])
        assert code == 0
        assert html_out.is_file()
        content = html_out.read_text(encoding="utf-8")
        assert "Timeline Board" in content or "canvas" in content
