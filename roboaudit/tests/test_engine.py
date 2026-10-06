"""End-to-end regression tests for the unified RoboAuditEngine orchestrator.

These tests exercise the full ``audit_episode`` pipeline on the bundled sample
episode. The component unit/property tests cover each analyzer in isolation, but
nothing exercised the end-to-end engine wiring -- a gap that previously allowed a
missing ``TimebaseIssue.to_data_issue`` method to crash the whole pipeline while
all unit tests still passed. This file locks that integration path in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roboaudit.core.models import AuditReport, QualityMetrics, Timeline
from roboaudit.engine import RoboAuditEngine

# Repository root -> sample_data/demo_episode (two parents up from this test file:
# roboaudit/tests/test_engine.py -> roboaudit/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_EPISODE = REPO_ROOT / "sample_data" / "demo_episode"


@pytest.fixture(scope="module")
def sample_report() -> AuditReport:
    """Run the full engine once on the sample episode and reuse the report."""
    if not SAMPLE_EPISODE.exists():
        pytest.skip(f"sample episode not found at {SAMPLE_EPISODE}")
    engine = RoboAuditEngine()
    return engine.audit_episode(SAMPLE_EPISODE)


def test_audit_episode_returns_audit_report(sample_report: AuditReport) -> None:
    """The engine returns a fully-formed AuditReport, not an exception."""
    assert isinstance(sample_report, AuditReport)


def test_report_has_all_required_sections(sample_report: AuditReport) -> None:
    """All top-level report sections required by Requirement 7.1 are present."""
    assert sample_report.schema_version == "1.0.0"
    assert sample_report.context is not None
    assert isinstance(sample_report.timeline, Timeline)
    assert sample_report.completion is not None
    assert sample_report.goal_alignment is not None
    assert isinstance(sample_report.data_issues, list)
    assert isinstance(sample_report.operator_mistakes, list)
    assert isinstance(sample_report.quality_metrics, QualityMetrics)


def test_quality_metrics_in_valid_ranges(sample_report: AuditReport) -> None:
    """Quality and goal-alignment scores stay within [0.0, 1.0]."""
    qm = sample_report.quality_metrics
    assert 0.0 <= qm.quality_score <= 1.0
    assert 0.0 <= qm.goal_alignment_score <= 1.0
    assert qm.total_anomalies >= 0
    assert qm.critical_issues >= 0


def test_data_issues_are_well_formed(sample_report: AuditReport) -> None:
    """Every emitted data issue has a valid category/severity (catches the
    TimebaseIssue.to_data_issue regression directly)."""
    valid_categories = {"timebase", "sensor", "camera", "grasp"}
    valid_severities = {"low", "medium", "high", "error"}
    for issue in sample_report.data_issues:
        assert issue.category in valid_categories
        assert issue.severity in valid_severities
        assert issue.t_s >= 0.0


def test_audit_episode_is_deterministic(sample_report: AuditReport) -> None:
    """Re-running the audit yields an identical quality score (deterministic
    engine, Requirement: no ML models / reproducible results)."""
    engine = RoboAuditEngine()
    second = engine.audit_episode(SAMPLE_EPISODE)
    assert second.quality_metrics.quality_score == sample_report.quality_metrics.quality_score
    assert len(second.data_issues) == len(sample_report.data_issues)
