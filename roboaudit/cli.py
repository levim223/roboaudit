"""Command-line interface for the RoboAudit engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboaudit.batch import BatchProcessor
from roboaudit.core.config import load_config
from roboaudit.engine import RoboAuditEngine
from roboaudit.reporting.generator import AuditParser, AuditPrettyPrinter, AuditReportGenerator
from roboaudit.visualization.board import TimelineBoardGenerator, serve_board


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roboaudit",
        description="Deterministic data quality and invariant auditing engine for robotics demonstration datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: audit
    audit_parser = subparsers.add_parser("audit", help="Audit a single demonstration episode")
    audit_parser.add_argument("episode_path", type=str, help="Path to episode directory or archive (.zip/.tar.gz)")
    audit_parser.add_argument("--config", "-c", type=str, default="", help="Path to audit config JSON file")
    audit_parser.add_argument("--profile", "-p", type=str, default="", help="Hardware profile name (umi, franka, kinova, generic)")
    audit_parser.add_argument("--output", "-o", type=str, default="", help="Path to write JSON audit report")
    audit_parser.add_argument("--markdown", "-m", type=str, default="", help="Path to write Markdown summary report")
    audit_parser.add_argument("--html", type=str, default="", help="Path to write HTML Timeline Board")
    audit_parser.add_argument("--json", action="store_true", help="Print JSON report to stdout")

    # Command: batch
    batch_parser = subparsers.add_parser("batch", help="Audit a directory of demonstration episodes in parallel")
    batch_parser.add_argument("episodes_dir", type=str, help="Path to directory containing episode subdirectories")
    batch_parser.add_argument("--config", "-c", type=str, default="", help="Path to audit config JSON file")
    batch_parser.add_argument("--workers", "-w", type=int, default=4, help="Worker threads for parallel processing (default 4)")
    batch_parser.add_argument("--output", "-o", type=str, default="", help="Path to write batch summary JSON")
    batch_parser.add_argument("--markdown", "-m", type=str, default="", help="Path to write batch summary Markdown")

    # Command: visualize
    vis_parser = subparsers.add_parser("visualize", help="Generate or serve an interactive Timeline Board")
    vis_parser.add_argument("episode_path", type=str, help="Path to episode directory or archive")
    vis_parser.add_argument("--config", "-c", type=str, default="", help="Path to audit config JSON file")
    vis_parser.add_argument("--output", "-o", type=str, default="", help="Path to write output HTML file")
    vis_parser.add_argument("--serve", action="store_true", help="Start local HTTP server to view the board")
    vis_parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default 8000)")

    # Command: validate
    val_parser = subparsers.add_parser("validate", help="Validate an audit report JSON file against schema v1.0.0")
    val_parser.add_argument("report_file", type=str, help="Path to audit report JSON file to validate")

    return parser


def handle_audit(args: argparse.Namespace) -> int:
    config = None
    if args.config or args.profile:
        config = load_config(source=args.config or None, robot_type=args.profile or None)

    engine = RoboAuditEngine(config=config)
    try:
        report = engine.audit_episode(args.episode_path)
    except Exception as exc:
        print(f"Error: audit failed: {exc}", file=sys.stderr)
        return 1

    generator = AuditReportGenerator()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(generator.export_json(report))
        print(f"JSON report written to: {out_path}")

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generator.export_markdown(report))
        print(f"Markdown report written to: {md_path}")

    if args.html:
        html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        board_gen = TimelineBoardGenerator()
        board_gen.generate_html(report, html_path)
        print(f"Timeline Board written to: {html_path}")

    if args.json:
        print(generator.to_json(report, indent=2))
    elif not (args.output or args.markdown or args.html):
        print("\n========================================================")
        print(f" RoboAudit Quality Report: {report.context.episode_id}")
        print("========================================================")
        print(f" Dataset:           {report.context.dataset}")
        print(f" Rig:               {report.context.rig}")
        print(f" Instruction:       {report.context.instruction}")
        print(f" Duration:          {report.context.length_s:.2f}s")
        print(f" Quality Score:     {report.quality_metrics.quality_score:.2f} / 1.00")
        print(f" Goal Alignment:    {report.quality_metrics.goal_alignment_score:.2f} ({report.goal_alignment.relation})")
        print(f" Task Completed:    {report.completion.task_completed}")
        print(f" Total Anomalies:   {report.quality_metrics.total_anomalies}")
        print(f" Data Issues:       {len(report.data_issues)}")
        print(f" Operator Mistakes: {len(report.operator_mistakes)}")
        if report.data_issues:
            print("\n Detected Data Issues:")
            for issue in report.data_issues:
                print(f"  - [{issue.severity.upper()}] {issue.category} @ {issue.t_s:.2f}s: {issue.issue}")
        if report.operator_mistakes:
            print("\n Detected Operator Mistakes:")
            for mistake in report.operator_mistakes:
                print(f"  - [{mistake.severity.upper()}] {mistake.type} @ {mistake.t_s:.2f}s: {mistake.evidence}")
        print("========================================================\n")

    return 0


def handle_batch(args: argparse.Namespace) -> int:
    config = None
    if args.config:
        config = load_config(source=args.config)

    root = Path(args.episodes_dir)
    if not root.is_dir():
        print(f"Error: {root} is not a valid directory", file=sys.stderr)
        return 1

    episode_paths = sorted([p for p in root.iterdir() if p.is_dir()])
    if not episode_paths:
        print(f"Error: no episode subdirectories found in {root}", file=sys.stderr)
        return 1

    processor = BatchProcessor(config=config)
    results, summary = processor.process_batch(episode_paths, max_workers=args.workers)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"Batch summary JSON written to: {out_path}")

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary.to_markdown())
        print(f"Batch summary Markdown written to: {md_path}")

    print("\nBatch Processing Summary:")
    print(f"  Episodes Processed: {summary.total_episodes}")
    print(f"  Pass Rate:          {summary.pass_rate * 100:.1f}%")
    print(f"  Mean Quality Score: {summary.mean_quality_score:.2f}")
    if summary.failed_episodes > 0:
        print(f"  Failed Episodes:    {summary.failed_episodes}")

    return 0


def handle_visualize(args: argparse.Namespace) -> int:
    config = None
    if args.config:
        config = load_config(source=args.config)

    engine = RoboAuditEngine(config=config)
    try:
        report = engine.audit_episode(args.episode_path)
    except Exception as exc:
        print(f"Error: failed to audit episode for visualization: {exc}", file=sys.stderr)
        return 1

    board_gen = TimelineBoardGenerator()
    out_path = Path(args.output) if args.output else Path(args.episode_path) / "roboaudit_timeline.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board_gen.generate_html(report, out_path)
    print(f"Timeline Board generated at: {out_path}")

    if args.serve:
        print(f"Serving Timeline Board on http://localhost:{args.port} (Ctrl+C to stop)...")
        serve_board(out_path, port=args.port)

    return 0


def handle_validate(args: argparse.Namespace) -> int:
    path = Path(args.report_file)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    parser = AuditParser()
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        report = parser.parse_report(content)
        print(f"VALID: Report conforms to schema v{report.schema_version}")
        print(f"  Episode ID:    {report.context.episode_id}")
        print(f"  Quality Score: {report.quality_metrics.quality_score:.2f}")
        print(f"  Data Issues:   {len(report.data_issues)}")
        print(f"  Mistakes:      {len(report.operator_mistakes)}")
        return 0
    except Exception as exc:
        print(f"INVALID: Schema validation failed: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "audit":
        return handle_audit(args)
    elif args.command == "batch":
        return handle_batch(args)
    elif args.command == "visualize":
        return handle_visualize(args)
    elif args.command == "validate":
        return handle_validate(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
