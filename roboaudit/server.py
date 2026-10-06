"""RoboAudit MCP server (Lesson 6 — Model Context Protocol).

Exposes the deterministic RoboAudit engine to Kiro (or any MCP client) over a
local stdio transport. The tool surface matches the "MCP Server Interface"
specified in ``.kiro/specs/roboaudit-engine/design.md`` for direct spec->code
traceability:

    - audit_episode     : audit a single episode, return the structured report
    - batch_audit       : audit every episode subdirectory, return an aggregate summary
    - visualize_timeline : render the interactive HTML timeline board for an episode
    - validate_report   : parse & schema-validate an audit report JSON string

Run standalone:        python -m roboaudit.server
Registered in Kiro via .kiro/settings/mcp.json (command: python, args: -m roboaudit.server).

Uses the official ``mcp`` package (``mcp.server.fastmcp.FastMCP``); no extra
dependency beyond what the project already provides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from roboaudit.batch import BatchProcessor
from roboaudit.engine import RoboAuditEngine
from roboaudit.reporting.generator import AuditParser, AuditReportGenerator
from roboaudit.visualization.board import TimelineBoardGenerator

mcp = FastMCP("roboaudit")

# Reusable engine/generator instances (deterministic, safe to share).
_engine = RoboAuditEngine()
_report_generator = AuditReportGenerator()
_parser = AuditParser()


def _audit_to_dict(episode_path: str) -> Dict[str, Any]:
    """Run the engine on one episode and return the report as a plain dict."""
    report = _engine.audit_episode(episode_path)
    return _report_generator.to_dict(report)


@mcp.tool()
def audit_episode(episode_path: str) -> Dict[str, Any]:
    """Audit a single robotics demonstration episode and return its quality report.

    Args:
        episode_path: Path to the episode directory or archive (containing video
            and/or telemetry) to audit.

    Returns:
        The structured audit report (schema_version, context, timeline,
        completion, goal_alignment, data_issues, operator_mistakes,
        quality_metrics) as a JSON-serializable object.
    """
    path = Path(episode_path)
    if not path.exists():
        return {"error": f"episode path does not exist: {episode_path}"}
    try:
        return _audit_to_dict(episode_path)
    except Exception as exc:  # surface a clean error to the MCP client
        return {"error": f"audit failed for {episode_path}: {exc}"}


@mcp.tool()
def batch_audit(episodes_dir: str, max_workers: int = 4) -> Dict[str, Any]:
    """Audit every episode subdirectory under a dataset directory in parallel.

    Args:
        episodes_dir: Directory containing one subdirectory per episode.
        max_workers: Number of parallel worker threads (default 4).

    Returns:
        An aggregate batch summary (counts, mean quality score, pass rate,
        per-episode scores, and any processing failures).
    """
    root = Path(episodes_dir)
    if not root.is_dir():
        return {"error": f"episodes_dir is not a directory: {episodes_dir}"}

    episode_paths: List[Path] = sorted(p for p in root.iterdir() if p.is_dir())
    if not episode_paths:
        return {"error": f"no episode subdirectories found in: {episodes_dir}"}

    try:
        processor = BatchProcessor()
        _results, summary = processor.process_batch(episode_paths, max_workers=max_workers)
        return summary.to_dict()
    except Exception as exc:
        return {"error": f"batch audit failed for {episodes_dir}: {exc}"}


@mcp.tool()
def visualize_timeline(episode_path: str, output_path: str = "") -> Dict[str, Any]:
    """Audit an episode and render its interactive HTML timeline board to disk.

    Args:
        episode_path: Path to the episode directory or archive to audit.
        output_path: Destination .html file path. Defaults to
            '<episode_path>/roboaudit_timeline.html' when left blank.

    Returns:
        An object with the written board path and the episode's quality score.
    """
    path = Path(episode_path)
    if not path.exists():
        return {"error": f"episode path does not exist: {episode_path}"}

    out = Path(output_path) if output_path else path / "roboaudit_timeline.html"
    try:
        report = _engine.audit_episode(episode_path)
        board = TimelineBoardGenerator()
        board.generate_html(report, out)
        return {
            "board_path": str(out),
            "quality_score": report.quality_metrics.quality_score,
            "episode_id": report.context.episode_id,
        }
    except Exception as exc:
        return {"error": f"visualization failed for {episode_path}: {exc}"}


@mcp.tool()
def validate_report(report_json: str) -> Dict[str, Any]:
    """Parse and schema-validate an audit report JSON string.

    Args:
        report_json: The JSON text of an audit report to validate.

    Returns:
        An object with 'valid' (bool) and either the parsed episode_id/
        schema_version or a descriptive 'error' message.
    """
    try:
        report = _parser.parse_report(report_json)
        return {
            "valid": True,
            "schema_version": report.schema_version,
            "episode_id": report.context.episode_id,
            "quality_score": report.quality_metrics.quality_score,
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def main() -> None:
    """Entry point: run the MCP server over stdio (how Kiro launches it)."""
    mcp.run()


if __name__ == "__main__":
    main()
