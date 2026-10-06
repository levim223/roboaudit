"""Tests for GoalAlignmentScorer and scoring constraints.

Covers:
- Property 55: Goal Alignment Score Range (Req 14.1)
- Property 57: Failure Outcome Alignment Constraint (Req 14.5)
- Unit tests for weighted deductions (Req 14.2), breakdown reporting (Req 14.3),
  and custom scoring functions (Req 14.4)
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roboaudit.analysis.alignment import GoalAlignmentResult, GoalAlignmentScorer
from roboaudit.core.models import (
    DataIssue,
    GoalAlignment,
    OperatorMistake,
    TemporalWindow,
    Timeline,
)


@st.composite
def alignment_scenario_strategy(draw):
    """Generate arbitrary episode scenario with outcome, mistakes, issues, and windows."""
    outcome = draw(st.sampled_from(["success", "failure", "partial", "success_then_undone"]))

    # Operator mistakes
    n_mistakes = draw(st.integers(min_value=0, max_value=5))
    mistakes = []
    for _ in range(n_mistakes):
        sev = draw(st.sampled_from(["low", "medium", "high"]))
        mtype = draw(st.sampled_from(["drop", "alignment_struggle", "hesitation", "fumble"]))
        mistakes.append(OperatorMistake(type=mtype, severity=sev, t_s=1.0, evidence=["f1"]))

    # Data issues
    n_issues = draw(st.integers(min_value=0, max_value=5))
    issues = []
    for _ in range(n_issues):
        sev = draw(st.sampled_from(["low", "medium", "high", "error"]))
        cat = draw(st.sampled_from(["timebase", "sensor", "camera", "grasp"]))
        issues.append(DataIssue(issue="issue", category=cat, severity=sev, t_s=1.0, evidence=["f1"]))

    # Timeline windows
    n_windows = draw(st.integers(min_value=0, max_value=5))
    windows = []
    for i in range(n_windows):
        contrib = draw(st.sampled_from(["advancing", "wasteful", "idle"]))
        phase = "idle" if contrib == "idle" else "manipulate"
        windows.append(
            TemporalWindow(
                start_s=float(i),
                end_s=float(i + 1),
                action_phase=phase,
                arm_attribution="right",
                contribution_type=contrib,
                completion_percentage=0.5,
            )
        )
    timeline = Timeline(windows=windows) if windows else None

    return outcome, timeline, mistakes, issues


class TestGoalAlignmentProperties:
    """Property-based tests for GoalAlignmentScorer."""

    @settings(max_examples=100)
    @given(scenario=alignment_scenario_strategy())
    def test_property_55_goal_alignment_score_range(self, scenario):
        """Property 55: Goal Alignment Score Range.
        
        Validates Requirements: 14.1.
        The goal_alignment score MUST always be bounded in [0.0, 1.0].
        """
        outcome, timeline, mistakes, issues = scenario
        scorer = GoalAlignmentScorer()
        result = scorer.calculate_goal_alignment(
            outcome=outcome,
            timeline=timeline,
            mistakes=mistakes,
            issues=issues,
        )

        assert isinstance(result, GoalAlignmentResult)
        assert 0.0 <= result.score <= 1.0

    @settings(max_examples=100)
    @given(scenario=alignment_scenario_strategy())
    def test_property_57_failure_outcome_alignment_constraint(self, scenario):
        """Property 57: Failure Outcome Alignment Constraint.
        
        Validates Requirements: 14.5.
        WHEN Task_Outcome is failure, THE goal_alignment score SHALL not exceed 0.5.
        """
        _, timeline, mistakes, issues = scenario
        scorer = GoalAlignmentScorer()
        result = scorer.calculate_goal_alignment(
            outcome="failure",
            timeline=timeline,
            mistakes=mistakes,
            issues=issues,
        )

        assert result.score <= 0.5


class TestGoalAlignmentUnit:
    """Unit tests for GoalAlignmentScorer."""

    def test_clean_successful_episode(self):
        """Clean successful episode has score 1.0 and aligned relation."""
        scorer = GoalAlignmentScorer()
        result = scorer.calculate_goal_alignment(outcome="success")

        assert result.score == 1.0
        assert result.matches_given is True
        assert result.relation == "aligned"
        assert len(result.deductions) == 0

    def test_deduction_breakdown_included(self):
        """Requirement 14.2, 14.3: deductions for mistakes and anomalies tracked in breakdown."""
        mistakes = [
            OperatorMistake(type="drop", severity="high", t_s=2.0, evidence=["f1"]),
        ]
        issues = [
            DataIssue(issue="jitter", category="timebase", severity="medium", t_s=1.0, evidence=["f1"]),
        ]
        scorer = GoalAlignmentScorer()
        result = scorer.calculate_goal_alignment(
            outcome="success", mistakes=mistakes, issues=issues
        )

        assert result.score < 1.0
        assert "operator_mistakes" in result.deductions
        assert "data_anomalies" in result.deductions

    def test_custom_scoring_function(self):
        """Requirement 14.4: Support custom goal_alignment scoring functions."""
        def custom_fn(outcome, timeline, mistakes, issues):
            return GoalAlignmentResult(
                score=0.92,
                matches_given=True,
                relation="aligned",
                deductions={"custom_rule": 0.08},
                note="Custom scorer evaluated",
            )

        scorer = GoalAlignmentScorer(custom_scorer=custom_fn)
        result = scorer.calculate_goal_alignment(outcome="success")

        assert result.score == 0.92
        assert "custom_rule" in result.deductions

    def test_custom_scorer_still_enforces_failure_cap(self):
        """Requirement 14.5 is strictly enforced even on custom scoring functions."""
        def generous_custom_fn(outcome, timeline, mistakes, issues):
            return GoalAlignmentResult(
                score=0.99,
                matches_given=True,
                relation="aligned",
                deductions={},
                note="Overly generous scorer",
            )

        scorer = GoalAlignmentScorer(custom_scorer=generous_custom_fn)
        result = scorer.calculate_goal_alignment(outcome="failure")

        assert result.score <= 0.5

    def test_to_goal_alignment_conversion(self):
        result = GoalAlignmentResult(
            score=0.88,
            matches_given=True,
            relation="aligned",
            deductions={},
            note="Well executed",
        )
        ga = result.to_goal_alignment()
        assert isinstance(ga, GoalAlignment)
        assert ga.matches_given is True
        assert ga.relation == "aligned"
        assert "0.88" in ga.note
