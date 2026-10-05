"""Property-based tests and unit tests for InvariantChecker.

Covers:
- Property 18: Outcome-Completion Consistency (Req 5.1)
- Property 16: Progress Monotonicity During Advancement (Req 4.5)
- Property 19: Undone Temporal Ordering (Req 5.2)
- Unit tests for all formal invariant rules.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roboaudit.analysis.invariants import InvariantChecker, InvariantViolation
from roboaudit.core.models import TaskOutcome, TemporalWindow, Timeline


def make_window(
    start_s: float,
    end_s: float,
    completion: float,
    contribution: str = "advancing",
    arm: str = "right",
    phase: str = "manipulate",
) -> TemporalWindow:
    """Helper to create TemporalWindow."""
    return TemporalWindow(
        start_s=start_s,
        end_s=end_s,
        action_phase=phase,
        arm_attribution=arm,
        contribution_type=contribution,
        completion_percentage=completion,
    )


class TestInvariantProperties:
    """Hypothesis property-based tests for formal timeline invariants (Lesson 4)."""

    @settings(max_examples=100)
    @given(
        outcome_result=st.sampled_from(["failure", "partial", "unclear"]),
        peak_progress=st.floats(min_value=1.0, max_value=1.0),
    )
    def test_property_18_outcome_completion_consistency(
        self, outcome_result: str, peak_progress: float
    ):
        """Property 18: WHEN outcome is failure/partial AND completion == 1.0 THEN violation is detected."""
        checker = InvariantChecker()
        outcome = TaskOutcome(outcome=outcome_result)
        timeline = Timeline(
            windows=[
                make_window(0.0, 1.0, 0.5, contribution="advancing"),
                make_window(1.0, 2.0, peak_progress, contribution="advancing"),
            ]
        )

        violation = checker.check_outcome_vs_completion(outcome, timeline)
        assert violation is not None
        assert violation.invariant_name == "outcome_vs_completion"
        assert violation.severity == "high"

    @settings(max_examples=100)
    @given(
        p0=st.floats(min_value=0.5, max_value=0.9),
        drop=st.floats(min_value=0.05, max_value=0.4),
    )
    def test_property_16_progress_monotonicity(self, p0: float, drop: float):
        """Property 16: WHEN progress decreases during advancing phases THEN violation is detected."""
        checker = InvariantChecker()
        p1 = max(0.0, p0 - drop)
        timeline = Timeline(
            windows=[
                make_window(0.0, 1.0, p0, contribution="advancing"),
                make_window(1.0, 2.0, p1, contribution="advancing"),
            ]
        )

        violations = checker.check_progress_monotonicity(timeline)
        assert len(violations) >= 1
        assert any(v.invariant_name == "progress_monotonicity" for v in violations)

    @settings(max_examples=100)
    @given(
        goal_s=st.floats(min_value=5.0, max_value=20.0),
        delta=st.floats(min_value=0.0, max_value=5.0),
    )
    def test_property_19_undone_temporal_ordering(self, goal_s: float, delta: float):
        """Property 19: WHEN outcome is success_then_undone AND undone <= goal THEN violation is detected."""
        checker = InvariantChecker()
        # Invert: undone_s <= goal_s
        undone_s = goal_s - delta
        outcome = TaskOutcome(
            outcome="success_then_undone",
            goal_reached_at_s=goal_s,
            undone_at_s=undone_s,
        )

        violation = checker.check_undone_timing(outcome)
        assert violation is not None
        assert violation.invariant_name == "undone_timing"


class TestInvariantCheckerUnit:
    """Unit tests for individual invariant check rules."""

    def test_progress_vs_outcome_success_valid(self):
        checker = InvariantChecker()
        outcome = TaskOutcome(outcome="success", completed_at_s=2.0)
        timeline = Timeline(
            windows=[
                make_window(0.0, 1.0, 0.5),
                make_window(1.0, 2.0, 1.0),
            ]
        )
        assert checker.check_progress_vs_outcome(outcome, timeline) is None

    def test_progress_vs_outcome_success_invalid(self):
        checker = InvariantChecker()
        outcome = TaskOutcome(outcome="success", completed_at_s=2.0)
        timeline = Timeline(
            windows=[
                make_window(0.0, 1.0, 0.4),
                make_window(1.0, 2.0, 0.6),  # < 0.95
            ]
        )
        v = checker.check_progress_vs_outcome(outcome, timeline)
        assert v is not None
        assert v.invariant_name == "progress_vs_outcome"

    def test_outcome_vs_alignment(self):
        checker = InvariantChecker()
        outcome = TaskOutcome(outcome="success")
        assert checker.check_outcome_vs_alignment(outcome, "aligned") is None

        v = checker.check_outcome_vs_alignment(outcome, "different")
        assert v is not None
        assert v.invariant_name == "outcome_vs_alignment"

    def test_idle_contribution_consistency(self):
        checker = InvariantChecker()
        # Window 1 is idle but increases progress from 0.4 to 0.7 -> violation
        timeline = Timeline(
            windows=[
                make_window(0.0, 1.0, 0.4, contribution="advancing"),
                make_window(1.0, 2.0, 0.7, contribution="idle"),
            ]
        )
        violations = checker.check_idle_contribution_consistency(timeline)
        assert len(violations) == 1
        assert violations[0].invariant_name == "idle_contribution_consistency"

    def test_time_past_end(self):
        checker = InvariantChecker()
        timeline = Timeline(
            windows=[
                make_window(0.0, 5.0, 0.5),
                make_window(5.0, 12.0, 1.0),  # exceeds episode duration of 10.0s
            ]
        )
        violations = checker.check_time_past_end(timeline, episode_duration_s=10.0)
        assert len(violations) == 1
        assert violations[0].invariant_name == "time_past_end"

    def test_verify_all_invariants_clean_episode(self):
        checker = InvariantChecker()
        outcome = TaskOutcome(outcome="success", completed_at_s=5.0, goal_reached_at_s=5.0)
        timeline = Timeline(
            windows=[
                make_window(0.0, 2.0, 0.4, contribution="advancing"),
                make_window(2.0, 3.0, 0.4, contribution="idle"),
                make_window(3.0, 5.0, 1.0, contribution="advancing"),
            ]
        )
        violations = checker.verify_all_invariants(
            outcome=outcome,
            timeline=timeline,
            episode_duration_s=5.0,
            alignment_relation="aligned",
        )
        assert violations == []
