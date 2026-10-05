"""Audit report generator, serializer, parser, and pretty-printer."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from roboaudit.core.models import (
    AuditReport,
    DataIssue,
    EpisodeContext,
    GoalAlignment,
    OperatorMistake,
    QualityMetrics,
    TaskCompletion,
    TemporalWindow,
    Timeline,
)
from roboaudit.core.schema import SCHEMA_VERSION, validate_audit_report


class AuditReportGenerator:
    """Generates schema-validated audit reports and formats for export."""

    def __init__(self, schema_version: str = SCHEMA_VERSION) -> None:
        self.schema_version = schema_version

    def calculate_quality_score(
        self,
        data_issues: List[DataIssue],
        operator_mistakes: List[OperatorMistake],
        goal_alignment: GoalAlignment,
    ) -> QualityMetrics:
        """Calculate weighted quality metrics based on detected issues."""
        total_anomalies = len(data_issues) + len(operator_mistakes)

        critical_count = sum(
            1 for i in data_issues if i.severity in ("high", "error")
        ) + sum(1 for m in operator_mistakes if m.severity in ("high", "error"))

        warning_count = sum(
            1 for i in data_issues if i.severity in ("low", "medium")
        ) + sum(1 for m in operator_mistakes if m.severity in ("low", "medium"))

        if goal_alignment.matches_given:
            alignment_score = 1.0
        elif goal_alignment.relation.lower() == "partial":
            alignment_score = 0.5
        else:
            alignment_score = 0.0

        # Base score 1.0; 25% penalty per critical issue, 5% per warning
        score = 1.0 - (critical_count * 0.25) - (warning_count * 0.05)
        # Weight in goal alignment
        score = score * 0.7 + alignment_score * 0.3
        score = max(0.0, min(1.0, score))

        return QualityMetrics(
            total_anomalies=total_anomalies,
            critical_issues=critical_count,
            warning_count=warning_count,
            goal_alignment_score=float(alignment_score),
            quality_score=float(score),
        )

    def generate_report(
        self,
        context: EpisodeContext,
        timeline: Timeline,
        completion: TaskCompletion,
        goal_alignment: GoalAlignment,
        data_issues: Optional[List[DataIssue]] = None,
        operator_mistakes: Optional[List[OperatorMistake]] = None,
    ) -> AuditReport:
        """Assemble and validate a complete AuditReport.
        
        Args:
            context: Episode metadata and instruction context.
            timeline: Temporal action phases.
            completion: Task completion verdict.
            goal_alignment: Goal alignment assessment.
            data_issues: List of detected data/hardware issues.
            operator_mistakes: List of operator execution errors.
            
        Returns:
            Validated AuditReport instance.
            
        Raises:
            ValueError: If report fails schema validation.
        """
        issues = data_issues or []
        mistakes = operator_mistakes or []
        metrics = self.calculate_quality_score(issues, mistakes, goal_alignment)

        report = AuditReport(
            schema_version=self.schema_version,
            context=context,
            timeline=timeline,
            completion=completion,
            goal_alignment=goal_alignment,
            data_issues=issues,
            operator_mistakes=mistakes,
            quality_metrics=metrics,
        )

        # Validate against JSON schema
        report_dict = self.to_dict(report)
        is_valid, error = validate_audit_report(report_dict)
        if not is_valid:
            raise ValueError(
                f"Generated audit report failed schema validation: {error}"
            )

        return report

    def to_dict(self, report: AuditReport) -> Dict[str, Any]:
        """Convert AuditReport dataclass to dictionary matching schema."""
        return {
            "schema_version": report.schema_version,
            "context": {
                "dataset": report.context.dataset,
                "rig": report.context.rig,
                "length_s": report.context.length_s,
                "instruction": report.context.instruction,
                "episode_id": report.context.episode_id,
            },
            "timeline": [
                {
                    "start_s": w.start_s,
                    "end_s": w.end_s,
                    "arm": w.arm_attribution,
                    "action": w.action_phase,
                    "object": w.object,
                    "contribution": w.contribution_type,
                    "progress": w.completion_percentage,
                }
                for w in report.timeline.windows
            ],
            "completion": {
                "task_completed": report.completion.task_completed,
                "goal_reached_at_s": report.completion.goal_reached_at_s,
                "undone_at_s": report.completion.undone_at_s,
                "completed_at_s": report.completion.completed_at_s,
                "reason": report.completion.reason,
            },
            "goal_alignment": {
                "matches_given": report.goal_alignment.matches_given,
                "relation": report.goal_alignment.relation,
                "note": report.goal_alignment.note,
            },
            "data_issues": [
                {
                    "issue": i.issue,
                    "category": i.category,
                    "severity": i.severity,
                    "t_s": i.t_s,
                    "evidence": i.evidence,
                }
                for i in report.data_issues
            ],
            "operator_mistakes": [
                {
                    "type": m.type,
                    "severity": m.severity,
                    "t_s": m.t_s,
                    "duration_s": m.duration_s,
                    "evidence": m.evidence,
                }
                for m in report.operator_mistakes
            ],
            "quality_metrics": {
                "total_anomalies": report.quality_metrics.total_anomalies,
                "critical_issues": report.quality_metrics.critical_issues,
                "warning_count": report.quality_metrics.warning_count,
                "goal_alignment_score": report.quality_metrics.goal_alignment_score,
                "quality_score": report.quality_metrics.quality_score,
            },
        }

    def export_json(self, report: AuditReport, indent: int = 2) -> str:
        """Serialize AuditReport to JSON string."""
        return json.dumps(self.to_dict(report), indent=indent)

    def export_markdown(self, report: AuditReport) -> str:
        """Format AuditReport as human-readable Markdown summary."""
        verdict = "PASSED" if report.completion.task_completed else "FAILED"
        score_pct = report.quality_metrics.quality_score * 100.0

        lines = [
            f"# RoboAudit Report: {report.context.episode_id}",
            f"**Dataset:** `{report.context.dataset}` | **Rig:** `{report.context.rig}` | **Duration:** {report.context.length_s:.1f}s",
            f"**Instruction:** *\"{report.context.instruction}\"*",
            "",
            "## Summary Verdict",
            f"- **Outcome:** **{verdict}** ({report.completion.reason})",
            f"- **Quality Score:** **{score_pct:.1f}/100**",
            f"- **Goal Alignment:** {'Matches Instruction' if report.goal_alignment.matches_given else 'Mismatch'} (`{report.goal_alignment.relation}`)",
            f"- **Anomalies Detected:** {report.quality_metrics.total_anomalies} ({report.quality_metrics.critical_issues} critical, {report.quality_metrics.warning_count} warnings)",
            "",
            "## Timeline Segmentation",
            "| Window (s) | Arm | Action | Contribution | Progress |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for w in report.timeline.windows:
            lines.append(
                f"| {w.start_s:.1f} - {w.end_s:.1f}s | {w.arm_attribution} | {w.action_phase} | {w.contribution_type} | {w.completion_percentage:.0%} |"
            )

        if report.data_issues:
            lines.extend([
                "",
                "## Hardware & Data Issues",
                "| Category | Severity | Timestamp | Description |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for i in report.data_issues:
                lines.append(f"| {i.category} | {i.severity.upper()} | {i.t_s:.2f}s | {i.issue} |")

        if report.operator_mistakes:
            lines.extend([
                "",
                "## Operator Mistakes",
                "| Type | Severity | Timestamp | Evidence |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for m in report.operator_mistakes:
                lines.append(f"| {m.type} | {m.severity.upper()} | {m.t_s:.2f}s | {', '.join(m.evidence) or 'N/A'} |")

        return "\n".join(lines)


class AuditParser:
    """Parses JSON strings into strongly typed AuditReport dataclasses."""

    def parse_report(self, json_str: str) -> AuditReport:
        """Parse JSON into AuditReport with schema validation.
        
        Args:
            json_str: Serialized JSON report.
            
        Returns:
            AuditReport instance.
            
        Raises:
            ValueError: If JSON is invalid or fails schema check.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e

        is_valid, error = validate_audit_report(data)
        if not is_valid:
            raise ValueError(f"Schema validation failed: {error}")

        ctx_data = data["context"]
        context = EpisodeContext(
            dataset=str(ctx_data["dataset"]),
            rig=str(ctx_data["rig"]),
            length_s=float(ctx_data["length_s"]),
            instruction=str(ctx_data["instruction"]),
            episode_id=str(ctx_data["episode_id"]),
        )

        windows: List[TemporalWindow] = []
        for w in data.get("timeline", []):
            windows.append(
                TemporalWindow(
                    start_s=float(w["start_s"]),
                    end_s=float(w["end_s"]),
                    action_phase=w["action"],
                    arm_attribution=w["arm"],
                    contribution_type=w["contribution"],
                    completion_percentage=float(w["progress"]),
                    object=w.get("object"),
                )
            )
        timeline = Timeline(windows=windows)

        comp_data = data["completion"]
        completion = TaskCompletion(
            task_completed=bool(comp_data["task_completed"]),
            goal_reached_at_s=(
                float(comp_data["goal_reached_at_s"])
                if comp_data.get("goal_reached_at_s") is not None
                else None
            ),
            undone_at_s=(
                float(comp_data["undone_at_s"])
                if comp_data.get("undone_at_s") is not None
                else None
            ),
            completed_at_s=(
                float(comp_data["completed_at_s"])
                if comp_data.get("completed_at_s") is not None
                else None
            ),
            reason=str(comp_data.get("reason", "")),
        )

        ga_data = data["goal_alignment"]
        goal_alignment = GoalAlignment(
            matches_given=bool(ga_data["matches_given"]),
            relation=str(ga_data["relation"]),
            note=str(ga_data.get("note", "")),
        )

        data_issues: List[DataIssue] = []
        for i in data.get("data_issues", []):
            data_issues.append(
                DataIssue(
                    issue=str(i["issue"]),
                    category=i["category"],
                    severity=i["severity"],
                    t_s=float(i["t_s"]),
                    evidence=list(i.get("evidence", [])),
                )
            )

        operator_mistakes: List[OperatorMistake] = []
        for m in data.get("operator_mistakes", []):
            operator_mistakes.append(
                OperatorMistake(
                    type=m["type"],
                    severity=m["severity"],
                    t_s=float(m["t_s"]),
                    duration_s=float(m["duration_s"]) if m.get("duration_s") is not None else None,
                    evidence=list(m.get("evidence", [])),
                )
            )

        qm_data = data["quality_metrics"]
        quality_metrics = QualityMetrics(
            total_anomalies=int(qm_data["total_anomalies"]),
            critical_issues=int(qm_data["critical_issues"]),
            warning_count=int(qm_data["warning_count"]),
            goal_alignment_score=float(qm_data["goal_alignment_score"]),
            quality_score=float(qm_data["quality_score"]),
        )

        return AuditReport(
            schema_version=str(data["schema_version"]),
            context=context,
            timeline=timeline,
            completion=completion,
            goal_alignment=goal_alignment,
            data_issues=data_issues,
            operator_mistakes=operator_mistakes,
            quality_metrics=quality_metrics,
        )


class AuditPrettyPrinter:
    """Formats AuditReport as formatted JSON."""

    def format_report(self, report: AuditReport, indent: int = 2) -> str:
        """Format an AuditReport instance into indented JSON string."""
        generator = AuditReportGenerator(schema_version=report.schema_version)
        return generator.export_json(report, indent=indent)
