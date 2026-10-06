"""Tests for OperatorMistakeClassifier and mistake summary aggregation.

Covers:
- Property 23: Drop Mistake Classification (Req 6.1)
- Property 27: Operator Mistake Metadata Completeness (Req 6.5)
- Unit tests for alignment struggles (Req 6.2), hesitations (Req 6.3), fumbles (Req 6.4),
  and summary aggregation (Req 6.6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roboaudit.analysis.grasp import GraspAnomaly
from roboaudit.analysis.operator_mistakes import OperatorMistakeClassifier
from roboaudit.core.models import AuditConfig, OperatorMistake, TemporalWindow, Timeline


@st.composite
def drop_episode_strategy(draw):
    """Generate episode data with an object drop event."""
    t_drop = draw(st.floats(min_value=0.5, max_value=60.0))
    via_anomaly = draw(st.booleans())

    if via_anomaly:
        anomaly = GraspAnomaly(
            anomaly_type="missed_drop",
            severity="high",
            timestamp_s=round(t_drop, 2),
            gripper_state="closed",
            visual_evidence=[f"camera_front:frame_{int(round(t_drop * 30))}"],
            details="Object dropped visually while gripper closed",
        )
        return None, [anomaly], t_drop
    else:
        # Telemetry with drop flag
        df = pd.DataFrame(
            {
                "timestamp": [round(t_drop, 2)],
                "object_dropped": [True],
            }
        )
        return df, None, t_drop


class TestOperatorMistakeClassifierProperties:
    """Property-based tests for OperatorMistakeClassifier."""

    @settings(max_examples=100)
    @given(data=drop_episode_strategy())
    def test_property_23_drop_mistake_classification(self, data):
        """Property 23: Drop Mistake Classification.
        
        Validates Requirements: 6.1.
        WHEN visual evidence shows an object dropping from the gripper,
        THE Audit_Engine SHALL create an operator_mistakes entry with
        type drop and high severity.
        """
        telemetry, anomalies, t_drop = data
        classifier = OperatorMistakeClassifier()
        mistakes = classifier.classify_mistakes(
            telemetry=telemetry,
            grasp_anomalies=anomalies,
        )

        assert len(mistakes) >= 1
        drop_mistake = next((m for m in mistakes if m.type == "drop"), None)
        assert drop_mistake is not None
        assert drop_mistake.type == "drop"
        assert drop_mistake.severity == "high"
        assert abs(drop_mistake.t_s - t_drop) < 0.05
        assert len(drop_mistake.evidence) > 0

    @settings(max_examples=100)
    @given(
        m_type=st.sampled_from(["drop", "alignment_struggle", "hesitation", "fumble"]),
        t_s=st.floats(min_value=0.0, max_value=120.0),
        duration_s=st.floats(min_value=2.1, max_value=10.0),
    )
    def test_property_27_operator_mistake_metadata_completeness(
        self, m_type: str, t_s: float, duration_s: float
    ):
        """Property 27: Operator Mistake Metadata Completeness.
        
        Validates Requirements: 6.5.
        ALL operator mistakes must include timestamp, severity, type,
        and camera/sensor evidence.
        """
        classifier = OperatorMistakeClassifier()

        if m_type == "hesitation":
            window = TemporalWindow(
                start_s=round(t_s, 2),
                end_s=round(t_s + duration_s, 2),
                action_phase="idle",
                arm_attribution="right",
                contribution_type="idle",
                completion_percentage=0.2,
            )
            timeline = Timeline(windows=[window])
            mistakes = classifier.classify_mistakes(timeline=timeline)
        elif m_type == "alignment_struggle":
            n_samples = int(duration_s * 30)
            timestamps = np.linspace(t_s, t_s + duration_s, n_samples)
            df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "alignment_error_deg": np.full(n_samples, 22.0),  # > 15 deg
                }
            )
            mistakes = classifier.classify_mistakes(telemetry=df)
        elif m_type == "drop":
            df = pd.DataFrame({"timestamp": [round(t_s, 2)], "drop": [True]})
            mistakes = classifier.classify_mistakes(telemetry=df)
        else:  # fumble
            n_samples = 40
            timestamps = np.linspace(t_s, t_s + 1.0, n_samples)
            # rapid oscillating velocity
            vels = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(n_samples)])
            df = pd.DataFrame({"timestamp": timestamps, "velocity": vels})
            mistakes = classifier.classify_mistakes(telemetry=df)

        for m in mistakes:
            assert isinstance(m, OperatorMistake)
            assert m.t_s >= 0.0
            assert m.severity in {"low", "medium", "high", "error"}
            assert m.type in {"drop", "alignment_struggle", "hesitation", "fumble", "collision"}
            assert len(m.evidence) > 0


class TestOperatorMistakeClassifierUnit:
    """Unit tests for OperatorMistakeClassifier."""

    def test_alignment_struggle_detection(self):
        """Test Requirement 6.2: angle error > 15 deg for > 2.0s -> medium severity."""
        timestamps = np.linspace(1.0, 3.5, 75)  # 2.5 seconds
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "alignment_error_deg": np.full(75, 20.0),
            }
        )
        classifier = OperatorMistakeClassifier()
        mistakes = classifier.classify_mistakes(telemetry=df)

        assert len(mistakes) == 1
        m = mistakes[0]
        assert m.type == "alignment_struggle"
        assert m.severity == "medium"
        assert m.t_s == 1.0
        assert m.duration_s is not None
        assert m.duration_s >= 2.0

    def test_hesitation_detection(self):
        """Test Requirement 6.3: idle > 2.0s during task execution -> low severity."""
        window = TemporalWindow(
            start_s=5.0,
            end_s=8.0,  # 3.0 seconds
            action_phase="idle",
            arm_attribution="left",
            contribution_type="idle",
            completion_percentage=0.5,
        )
        timeline = Timeline(windows=[window])
        classifier = OperatorMistakeClassifier()
        mistakes = classifier.classify_mistakes(timeline=timeline)

        assert len(mistakes) == 1
        m = mistakes[0]
        assert m.type == "hesitation"
        assert m.severity == "low"
        assert m.t_s == 5.0

    def test_fumble_detection(self):
        """Test Requirement 6.4: trajectory oscillation > 3 reversals/s -> medium severity."""
        timestamps = np.linspace(0.0, 1.0, 30)
        # 10 direction reversals within 1.0s (> 3 reversals/s)
        vels = np.sin(np.linspace(0, 10 * np.pi, 30))
        df = pd.DataFrame({"timestamp": timestamps, "velocity": vels})

        classifier = OperatorMistakeClassifier()
        mistakes = classifier.classify_mistakes(telemetry=df)

        assert len(mistakes) >= 1
        fumble = next((m for m in mistakes if m.type == "fumble"), None)
        assert fumble is not None
        assert fumble.severity == "medium"

    def test_aggregate_mistakes_summary(self):
        """Test Requirement 6.6: Aggregate operator_mistakes into summary report."""
        mistakes = [
            OperatorMistake(type="drop", severity="high", t_s=1.0, evidence=["f1"]),
            OperatorMistake(type="hesitation", severity="low", t_s=4.0, evidence=["f2"]),
            OperatorMistake(type="fumble", severity="medium", t_s=7.0, evidence=["f3"]),
        ]
        classifier = OperatorMistakeClassifier()
        summary = classifier.aggregate_mistakes_summary(mistakes)

        assert summary["total_mistakes"] == 3
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["low"] == 1
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_type"]["drop"] == 1
        assert summary["by_type"]["hesitation"] == 1
        assert summary["by_type"]["fumble"] == 1
