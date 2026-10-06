"""Goal Alignment Scoring module for quantifying demonstration quality.

Calculates:
- Goal alignment score in [0.0, 1.0] (Req 14.1)
- Weighted deductions for wasteful segments, operator mistakes, and data anomalies (Req 14.2)
- Breakdown of deduction sources included in audit result (Req 14.3)
- Custom scoring function support via configuration (Req 14.4)
- Strict constraint: when outcome is failure, score cannot exceed 0.5 (Req 14.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from roboaudit.core.models import (
    DataIssue,
    GoalAlignment,
    OperatorMistake,
    Timeline,
)


@dataclass
class GoalAlignmentResult:
    """Result of goal alignment assessment with deduction breakdown.
    
    Attributes:
        score: Score in range [0.0, 1.0]
        matches_given: Whether demonstration matches instruction
        relation: Semantic relationship ('aligned', 'partial', 'different', 'unrelated')
        deductions: Breakdown of deduction sources and deducted values
        note: Descriptive explanatory summary
    """
    score: float
    matches_given: bool
    relation: str
    deductions: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_goal_alignment(self) -> GoalAlignment:
        """Convert to standard GoalAlignment dataclass."""
        return GoalAlignment(
            matches_given=self.matches_given,
            relation=self.relation,
            note=f"{self.note} (score: {self.score:.2f})",
        )


class GoalAlignmentScorer:
    """Quantifies demonstration alignment with task goals using weighted deductions."""

    def __init__(
        self,
        custom_scorer: Optional[Callable[[str, Optional[Timeline], Optional[List[OperatorMistake]], Optional[List[DataIssue]]], GoalAlignmentResult]] = None,
    ):
        """Initialize scorer with optional custom scoring function.
        
        Args:
            custom_scorer: Optional custom scoring function (Req 14.4).
        """
        self.custom_scorer = custom_scorer

    def calculate_goal_alignment(
        self,
        outcome: str,
        timeline: Optional[Timeline] = None,
        mistakes: Optional[List[OperatorMistake]] = None,
        issues: Optional[List[DataIssue]] = None,
    ) -> GoalAlignmentResult:
        """Calculate goal alignment score and deduction breakdown.
        
        Requirements 14.1, 14.2, 14.3, 14.5.
        
        Args:
            outcome: Task outcome ('success', 'failure', 'partial', 'success_then_undone').
            timeline: Optional Timeline with TemporalWindows.
            mistakes: Optional list of detected OperatorMistake instances.
            issues: Optional list of detected DataIssue instances.
            
        Returns:
            GoalAlignmentResult with score in [0.0, 1.0], deduction breakdown, and notes.
        """
        if self.custom_scorer is not None:
            result = self.custom_scorer(outcome, timeline, mistakes, issues)
            # Enforce Requirement 14.5 even on custom scorer
            if outcome == "failure":
                result.score = min(0.5, result.score)
            result.score = max(0.0, min(1.0, result.score))
            return result

        score = 1.0
        deductions: Dict[str, float] = {}

        # 1. Deductions for wasteful temporal windows (Requirement 14.2)
        wasteful_count = 0
        if timeline and timeline.windows:
            for w in timeline.windows:
                if w.contribution_type == "wasteful":
                    wasteful_count += 1
        if wasteful_count > 0:
            deduction = round(min(0.25, wasteful_count * 0.05), 3)
            deductions["wasteful_segments"] = deduction
            score -= deduction

        # 2. Deductions for operator mistakes (Requirement 14.2)
        mistake_deduction = 0.0
        if mistakes:
            for m in mistakes:
                if m.severity == "high":
                    mistake_deduction += 0.15
                elif m.severity == "medium":
                    mistake_deduction += 0.08
                else:  # low
                    mistake_deduction += 0.03
        if mistake_deduction > 0.0:
            mistake_deduction = round(min(0.40, mistake_deduction), 3)
            deductions["operator_mistakes"] = mistake_deduction
            score -= mistake_deduction

        # 3. Deductions for data issues (Requirement 14.2)
        issue_deduction = 0.0
        if issues:
            for i in issues:
                if i.severity in {"error", "high"}:
                    issue_deduction += 0.10
                elif i.severity == "medium":
                    issue_deduction += 0.05
                else:
                    issue_deduction += 0.02
        if issue_deduction > 0.0:
            issue_deduction = round(min(0.30, issue_deduction), 3)
            deductions["data_anomalies"] = issue_deduction
            score -= issue_deduction

        # 4. Outcome penalties and constraints
        if outcome == "partial":
            score = min(0.75, score)
            deductions["partial_outcome_cap"] = 0.25
        elif outcome == "failure":
            # Requirement 14.5: WHEN Task_Outcome is failure, THE goal_alignment score SHALL not exceed 0.5
            if score > 0.5:
                deductions["failure_outcome_cap"] = round(score - 0.5, 3)
                score = 0.5

        # Clamp score to [0.0, 1.0] (Requirement 14.1)
        final_score = max(0.0, min(1.0, round(score, 3)))

        # Determine relation
        if final_score >= 0.85 and outcome == "success":
            relation = "aligned"
            matches_given = True
        elif final_score >= 0.50:
            relation = "partial"
            matches_given = outcome in {"success", "partial"}
        elif final_score >= 0.25:
            relation = "different"
            matches_given = False
        else:
            relation = "unrelated"
            matches_given = False

        note = f"Goal alignment evaluated with {len(deductions)} deduction categories"

        return GoalAlignmentResult(
            score=final_score,
            matches_given=matches_given,
            relation=relation,
            deductions=deductions,
            note=note,
        )
