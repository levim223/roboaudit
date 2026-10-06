"""Tests for TimelineBoardGenerator and serve_board HTTP server.

Covers:
- Task 18.1: HTML generation with complete multi-track layout
- Task 18.2: Interactive features, color-coding, and progress bar
- Task 18.3: Lightweight HTTP server serving timeline board
"""

from __future__ import annotations

from pathlib import Path
import urllib.request
import pytest

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
from roboaudit.visualization.board import TimelineBoardGenerator, serve_board


@pytest.fixture
def sample_audit_report() -> AuditReport:
    """Fixture providing a complete, realistic AuditReport."""
    context = EpisodeContext(
        dataset="teleop_eval",
        rig="umi_handheld",
        length_s=12.5,
        instruction="pick the green mug and place it on coaster",
        episode_id="ep_sample_01",
    )
    windows = [
        TemporalWindow(
            start_s=0.0,
            end_s=3.0,
            action_phase="approach",
            arm_attribution="right",
            contribution_type="advancing",
            completion_percentage=0.25,
        ),
        TemporalWindow(
            start_s=3.0,
            end_s=5.0,
            action_phase="grasp",
            arm_attribution="right",
            contribution_type="advancing",
            completion_percentage=0.50,
        ),
        TemporalWindow(
            start_s=5.0,
            end_s=7.0,
            action_phase="idle",
            arm_attribution="none",
            contribution_type="idle",
            completion_percentage=0.50,
        ),
        TemporalWindow(
            start_s=7.0,
            end_s=9.0,
            action_phase="manipulate",
            arm_attribution="right",
            contribution_type="wasteful",
            completion_percentage=0.50,
        ),
        TemporalWindow(
            start_s=9.0,
            end_s=12.5,
            action_phase="release",
            arm_attribution="right",
            contribution_type="advancing",
            completion_percentage=1.00,
        ),
    ]
    completion = TaskCompletion(
        task_completed=True,
        goal_reached_at_s=12.5,
        undone_at_s=None,
        completed_at_s=12.5,
        reason="Mug successfully placed on coaster",
    )
    goal_alignment = GoalAlignment(
        matches_given=True,
        relation="aligned",
        note="Demonstration matches instruction",
    )
    data_issues = [
        DataIssue(
            issue="timebase_jitter: frame interval jitter exceeded 2%",
            category="timebase",
            severity="medium",
            t_s=4.2,
            evidence=["frame_126"],
        ),
    ]
    operator_mistakes = [
        OperatorMistake(
            type="hesitation",
            severity="low",
            t_s=5.0,
            duration_s=2.0,
            evidence=["frame_150"],
        ),
    ]
    metrics = QualityMetrics(
        total_anomalies=2,
        critical_issues=0,
        warning_count=2,
        goal_alignment_score=0.92,
        quality_score=0.88,
    )
    return AuditReport(
        schema_version="1.0.0",
        context=context,
        timeline=Timeline(windows=windows),
        completion=completion,
        goal_alignment=goal_alignment,
        data_issues=data_issues,
        operator_mistakes=operator_mistakes,
        quality_metrics=metrics,
    )


class TestTimelineBoard:
    """Test suite for timeline board generation and serving."""

    def test_generate_html_contains_required_tracks_and_data(self, sample_audit_report):
        """Test Requirement 9.1, 9.2: HTML generated contains tracks, metadata, and JSON."""
        gen = TimelineBoardGenerator()
        html = gen.generate_html(sample_audit_report)

        assert "<!DOCTYPE html>" in html
        assert "RoboAudit Timeline Board" in html
        assert "ep_sample_01" in html
        assert "pick the green mug and place it on coaster" in html
        assert "phaseTrack" in html
        assert "anomalyTrack" in html
        assert "progressFill" in html
        assert "timeDisplay" in html

    def test_color_coding_classes_present(self, sample_audit_report):
        """Test Requirement 9.5: Color coding for advancing, wasteful, idle."""
        gen = TimelineBoardGenerator()
        html = gen.generate_html(sample_audit_report)

        assert "phase-advancing" in html
        assert "phase-wasteful" in html
        assert "phase-idle" in html

    def test_save_html_file(self, sample_audit_report, tmp_path: Path):
        """Test saving HTML board to disk."""
        gen = TimelineBoardGenerator()
        out_file = tmp_path / "board.html"
        saved_path = gen.save_html(sample_audit_report, out_file)

        assert saved_path.is_file()
        content = saved_path.read_text(encoding="utf-8")
        assert len(content) > 500

    def test_serve_board_http(self, sample_audit_report):
        """Test Requirement 9.1: Lightweight HTTP server serves board with status 200."""
        gen = TimelineBoardGenerator()
        html = gen.generate_html(sample_audit_report)

        server = serve_board(html, port=8899)
        try:
            req = urllib.request.Request("http://127.0.0.1:8899")
            with urllib.request.urlopen(req, timeout=3.0) as response:
                assert response.status == 200
                data = response.read().decode("utf-8")
                assert "RoboAudit Timeline Board" in data
        finally:
            server.shutdown()
            server.server_close()
