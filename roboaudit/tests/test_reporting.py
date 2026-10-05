"""Property-based tests and unit tests for AuditReportGenerator and AuditParser.

Covers:
- Property 34: Report Serialization Round-Trip (Req 8.4)
- Property 28: Audit Report Structure Completeness (Req 7.1)
- Unit tests for Markdown export and quality score calculation.
"""

from __future__ import annotations

import json
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
from roboaudit.core.schema import SCHEMA_VERSION
from roboaudit.reporting.generator import (
    AuditParser,
    AuditPrettyPrinter,
    AuditReportGenerator,
)


@st.composite
def audit_report_strategy(draw):
    """Hypothesis strategy generating valid AuditReport instances."""
    dataset = draw(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"))
    rig = draw(st.sampled_from(["teleop_arm", "bimanual_arms", "handheld_gripper"]))
    length_s = draw(st.floats(min_value=5.0, max_value=120.0))
    instruction = draw(st.text(min_size=5, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz "))
    episode_id = f"ep_{draw(st.integers(min_value=1, max_value=9999))}"

    context = EpisodeContext(
        dataset=dataset,
        rig=rig,
        length_s=length_s,
        instruction=instruction,
        episode_id=episode_id,
    )

    num_windows = draw(st.integers(min_value=1, max_value=5))
    windows = []
    t_start = 0.0
    for i in range(num_windows):
        duration = draw(st.floats(min_value=1.0, max_value=5.0))
        t_end = min(length_s, t_start + duration)
        if t_end <= t_start:
            t_end = t_start + 1.0
        progress = min(1.0, float(i + 1) / float(num_windows))
        arm = draw(st.sampled_from(["left", "right", "both", "none"]))
        phase = draw(st.sampled_from(["approach", "grasp", "manipulate", "release", "idle"]))
        contrib = draw(st.sampled_from(["advancing", "wasteful", "idle"]))

        windows.append(
            TemporalWindow(
                start_s=round(t_start, 2),
                end_s=round(t_end, 2),
                action_phase=phase,
                arm_attribution=arm,
                contribution_type=contrib,
                completion_percentage=round(progress, 2),
            )
        )
        t_start = t_end

    timeline = Timeline(windows=windows)

    task_completed = draw(st.booleans())
    completion = TaskCompletion(
        task_completed=task_completed,
        goal_reached_at_s=round(length_s * 0.8, 2) if task_completed else None,
        undone_at_s=None,
        completed_at_s=round(length_s * 0.8, 2) if task_completed else None,
        reason="Demonstration completed successfully" if task_completed else "Task failed",
    )

    matches_given = draw(st.booleans())
    relation = "aligned" if matches_given else draw(st.sampled_from(["different", "unrelated", "partial"]))
    goal_alignment = GoalAlignment(
        matches_given=matches_given,
        relation=relation,
        note="Goal alignment evaluated against instruction",
    )

    # Data issues
    num_issues = draw(st.integers(min_value=0, max_value=3))
    data_issues = []
    for _ in range(num_issues):
        cat = draw(st.sampled_from(["timebase", "sensor", "camera", "grasp"]))
        sev = draw(st.sampled_from(["low", "medium", "high", "error"]))
        t_s = draw(st.floats(min_value=0.0, max_value=length_s))
        data_issues.append(
            DataIssue(
                issue="Test detected issue",
                category=cat,
                severity=sev,
                t_s=round(t_s, 2),
                evidence=["frame_10"],
            )
        )

    # Operator mistakes
    num_mistakes = draw(st.integers(min_value=0, max_value=2))
    operator_mistakes = []
    for _ in range(num_mistakes):
        mtype = draw(st.sampled_from(["drop", "alignment_struggle", "hesitation", "fumble"]))
        msev = draw(st.sampled_from(["low", "medium", "high"]))
        mt_s = draw(st.floats(min_value=0.0, max_value=length_s))
        operator_mistakes.append(
            OperatorMistake(
                type=mtype,
                severity=msev,
                t_s=round(mt_s, 2),
                evidence=["frame_15"],
            )
        )

    generator = AuditReportGenerator()
    metrics = generator.calculate_quality_score(data_issues, operator_mistakes, goal_alignment)

    return AuditReport(
        schema_version=SCHEMA_VERSION,
        context=context,
        timeline=timeline,
        completion=completion,
        goal_alignment=goal_alignment,
        data_issues=data_issues,
        operator_mistakes=operator_mistakes,
        quality_metrics=metrics,
    )


class TestReportingProperties:
    """Hypothesis property-based tests for report serialization and completeness (Lesson 4)."""

    @settings(max_examples=100)
    @given(report=audit_report_strategy())
    def test_property_34_serialization_round_trip(self, report: AuditReport):
        """Property 34: For any valid AuditReport, parse(print(report)) produces equivalent report."""
        printer = AuditPrettyPrinter()
        parser = AuditParser()

        # Print to JSON
        json_str = printer.format_report(report)

        # Parse back
        parsed_report = parser.parse_report(json_str)

        # Compare equivalence
        assert parsed_report.schema_version == report.schema_version
        assert parsed_report.context.episode_id == report.context.episode_id
        assert parsed_report.context.dataset == report.context.dataset
        assert parsed_report.completion.task_completed == report.completion.task_completed
        assert parsed_report.goal_alignment.matches_given == report.goal_alignment.matches_given
        assert len(parsed_report.timeline.windows) == len(report.timeline.windows)
        assert len(parsed_report.data_issues) == len(report.data_issues)
        assert len(parsed_report.operator_mistakes) == len(report.operator_mistakes)
        assert abs(parsed_report.quality_metrics.quality_score - report.quality_metrics.quality_score) < 1e-4

    @settings(max_examples=100)
    @given(report=audit_report_strategy())
    def test_property_28_report_structure_completeness(self, report: AuditReport):
        """Property 28: Emitted JSON must contain all required top-level sections."""
        generator = AuditReportGenerator()
        report_dict = generator.to_dict(report)

        required_keys = {
            "schema_version",
            "context",
            "timeline",
            "completion",
            "goal_alignment",
            "data_issues",
            "operator_mistakes",
            "quality_metrics",
        }

        assert required_keys.issubset(report_dict.keys())


class TestAuditReportGeneratorUnit:
    """Unit tests for report generation, Markdown formatting, and score calculation."""

    def test_markdown_export(self):
        context = EpisodeContext("molmo", "teleop_arms", 10.0, "pick up cube", "ep_001")
        timeline = Timeline(
            windows=[
                TemporalWindow(0.0, 5.0, "approach", "right", "advancing", 0.5),
                TemporalWindow(5.0, 10.0, "grasp", "right", "advancing", 1.0),
            ]
        )
        completion = TaskCompletion(True, 10.0, None, 10.0, "Cube grasped")
        goal_alignment = GoalAlignment(True, "aligned", "Matches cube pickup")

        generator = AuditReportGenerator()
        report = generator.generate_report(context, timeline, completion, goal_alignment)

        md = generator.export_markdown(report)
        assert "# RoboAudit Report: ep_001" in md
        assert "PASSED" in md
        assert "Cube grasped" in md
        assert "| 0.0 - 5.0s |" in md

    def test_parser_invalid_json(self):
        parser = AuditParser()
        with pytest.raises(ValueError, match="Invalid JSON string"):
            parser.parse_report("{invalid_json: true")

    def test_parser_unsupported_version(self):
        parser = AuditParser()
        bad_report = {
            "schema_version": "99.0.0",
            "context": {"dataset": "d", "rig": "r", "length_s": 1.0, "instruction": "i", "episode_id": "e"},
            "timeline": [],
            "completion": {"task_completed": True, "goal_reached_at_s": 1.0, "reason": "ok"},
            "goal_alignment": {"matches_given": True, "relation": "aligned"},
            "data_issues": [],
            "operator_mistakes": [],
            "quality_metrics": {
                "total_anomalies": 0,
                "critical_issues": 0,
                "warning_count": 0,
                "goal_alignment_score": 1.0,
                "quality_score": 1.0,
            },
        }
        with pytest.raises(ValueError, match="Unsupported schema version"):
            parser.parse_report(json.dumps(bad_report))
