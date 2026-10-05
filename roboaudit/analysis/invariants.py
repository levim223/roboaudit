"""Formal mathematical invariant checking for robotics demonstration timelines and outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from roboaudit.core.models import TaskOutcome, Timeline


@dataclass
class InvariantViolation:
    """Represents a violation of a formal invariant rule in timeline or outcome."""

    invariant_name: str
    severity: Literal["low", "medium", "high", "error"]
    details: str
    timestamp_s: Optional[float] = None


class InvariantChecker:
    """Pure functional validator for formal robotics demonstration invariants."""

    def check_outcome_vs_completion(
        self, outcome: TaskOutcome, timeline: Timeline
    ) -> Optional[InvariantViolation]:
        """Verify that failure or partial outcomes do not reach 100% completion.
        
        Rule: (outcome in ('failure', 'partial', 'unclear')) -> peak_completion < 1.0
        """
        res = getattr(outcome, "outcome", getattr(outcome, "result", ""))
        if res in ("failure", "partial", "unclear"):
            peak = max((w.completion_percentage for w in timeline.windows), default=0.0)
            if peak >= 0.999:
                return InvariantViolation(
                    invariant_name="outcome_vs_completion",
                    severity="high",
                    details=(
                        f"Task outcome is '{res}', but timeline completion reaches "
                        f"{peak:.1%}, which is reserved exclusively for successful outcomes."
                    ),
                )
        return None

    def check_progress_vs_outcome(
        self, outcome: TaskOutcome, timeline: Timeline, threshold: float = 0.95
    ) -> Optional[InvariantViolation]:
        """Verify that successful outcomes achieve at least 95% completion.
        
        Rule: outcome == 'success' -> final_completion >= 0.95
        """
        res = getattr(outcome, "outcome", getattr(outcome, "result", ""))
        if res == "success" and timeline.windows:
            final_progress = timeline.windows[-1].completion_percentage
            peak_progress = max(w.completion_percentage for w in timeline.windows)
            effective_progress = max(final_progress, peak_progress)

            if effective_progress < threshold:
                return InvariantViolation(
                    invariant_name="progress_vs_outcome",
                    severity="high",
                    details=(
                        f"Task outcome is 'success', but peak progress reaches only "
                        f"{effective_progress:.1%} (expected >= {threshold:.0%})."
                    ),
                )
        return None

    def check_outcome_vs_alignment(
        self, outcome: TaskOutcome, alignment_relation: str
    ) -> Optional[InvariantViolation]:
        """Verify that successful outcome aligns with instructed goal.
        
        Rule: outcome == 'success' requires alignment_relation == 'aligned'
        """
        relation = alignment_relation.lower().strip()
        res = getattr(outcome, "outcome", getattr(outcome, "result", ""))
        if res in ("success", "success_then_undone") and relation in (
            "different",
            "unrelated",
        ):
            return InvariantViolation(
                invariant_name="outcome_vs_alignment",
                severity="high",
                details=(
                    f"Task outcome is '{res}', but goal alignment indicates the demonstration "
                    f"depicted a '{relation}' task."
                ),
            )
        return None

    def check_undone_timing(self, outcome: TaskOutcome) -> Optional[InvariantViolation]:
        """Verify temporal ordering when a successful task was undone.
        
        Rule: outcome == 'success_then_undone' requires undone_at_s > goal_reached_at_s
        """
        res = getattr(outcome, "outcome", getattr(outcome, "result", ""))
        if res == "success_then_undone":
            if outcome.goal_reached_at_s is None or outcome.undone_at_s is None:
                return InvariantViolation(
                    invariant_name="undone_timing",
                    severity="high",
                    details="Outcome is 'success_then_undone' but goal_reached_at_s or undone_at_s is missing.",
                )
            if outcome.undone_at_s <= outcome.goal_reached_at_s:
                return InvariantViolation(
                    invariant_name="undone_timing",
                    severity="high",
                    details=(
                        f"Outcome is 'success_then_undone', but undo time ({outcome.undone_at_s:.2f}s) "
                        f"does not strictly exceed goal time ({outcome.goal_reached_at_s:.2f}s)."
                    ),
                    timestamp_s=outcome.undone_at_s,
                )
        return None

    def check_time_past_end(
        self, timeline: Timeline, episode_duration_s: float
    ) -> List[InvariantViolation]:
        """Verify that no event timestamps exceed the total episode duration."""
        violations: List[InvariantViolation] = []
        tolerance = 1.0  # 1.0s grace period for container rounding
        max_allowed = episode_duration_s + tolerance

        for idx, w in enumerate(timeline.windows):
            if w.end_s > max_allowed:
                violations.append(
                    InvariantViolation(
                        invariant_name="time_past_end",
                        severity="medium",
                        details=(
                            f"Timeline window {idx} end time ({w.end_s:.2f}s) exceeds episode "
                            f"duration ({episode_duration_s:.2f}s)."
                        ),
                        timestamp_s=w.end_s,
                    )
                )
        return violations

    def check_progress_monotonicity(self, timeline: Timeline) -> List[InvariantViolation]:
        """Verify that progress is non-decreasing during advancing phases.
        
        Rule: For advancing window i and advancing window j > i, progress(j) >= progress(i)
        """
        violations: List[InvariantViolation] = []
        last_advancing_progress: Optional[float] = None
        last_advancing_idx: Optional[int] = None

        for idx, w in enumerate(timeline.windows):
            if w.contribution_type == "advancing":
                if last_advancing_progress is not None:
                    if w.completion_percentage < last_advancing_progress - 1e-6:
                        violations.append(
                            InvariantViolation(
                                invariant_name="progress_monotonicity",
                                severity="high",
                                details=(
                                    f"Progress decreased from {last_advancing_progress:.1%} at window {last_advancing_idx} "
                                    f"to {w.completion_percentage:.1%} at window {idx} during active advancement."
                                ),
                                timestamp_s=w.start_s,
                            )
                        )
                last_advancing_progress = w.completion_percentage
                last_advancing_idx = idx

        return violations

    def check_idle_contribution_consistency(self, timeline: Timeline) -> List[InvariantViolation]:
        """Verify that idle windows do not increase task completion progress.
        
        Rule: During window with contribution_type == 'idle', progress cannot increase.
        """
        violations: List[InvariantViolation] = []
        current_level = 0.0

        for idx, w in enumerate(timeline.windows):
            if idx == 0:
                current_level = w.completion_percentage
                continue

            prev_w = timeline.windows[idx - 1]
            if w.contribution_type == "idle":
                # Idle step must not raise progress above previous level
                if w.completion_percentage > prev_w.completion_percentage + 1e-6:
                    violations.append(
                        InvariantViolation(
                            invariant_name="idle_contribution_consistency",
                            severity="high",
                            details=(
                                f"Window {idx} is classified as 'idle', but increases completion "
                                f"from {prev_w.completion_percentage:.1%} to {w.completion_percentage:.1%}."
                            ),
                            timestamp_s=w.start_s,
                        )
                    )
            current_level = w.completion_percentage

        return violations

    def verify_all_invariants(
        self,
        outcome: TaskOutcome,
        timeline: Timeline,
        episode_duration_s: float,
        alignment_relation: str = "aligned",
    ) -> List[InvariantViolation]:
        """Run all formal invariant checks and aggregate detected violations."""
        violations: List[InvariantViolation] = []

        # 1. Outcome vs Completion
        v = self.check_outcome_vs_completion(outcome, timeline)
        if v:
            violations.append(v)

        # 2. Progress vs Outcome
        v = self.check_progress_vs_outcome(outcome, timeline)
        if v:
            violations.append(v)

        # 3. Outcome vs Alignment
        v = self.check_outcome_vs_alignment(outcome, alignment_relation)
        if v:
            violations.append(v)

        # 4. Undone Timing
        v = self.check_undone_timing(outcome)
        if v:
            violations.append(v)

        # 5. Time past end
        violations.extend(self.check_time_past_end(timeline, episode_duration_s))

        # 6. Monotonicity
        violations.extend(self.check_progress_monotonicity(timeline))

        # 7. Idle consistency
        violations.extend(self.check_idle_contribution_consistency(timeline))

        return violations
