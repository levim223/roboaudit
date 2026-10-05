"""Tests for ActionPhaseSegmenter and operator hesitation detection.

Covers:
- Property 15: Temporal Window Classification Completeness (Req 4.1, 4.2, 4.3, 4.4)
- Property 17: Operator Hesitation Detection (Req 4.7)
- Unit tests for phase classification, idle behavior, and edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roboaudit.analysis.segmenter import ActionPhaseSegmenter
from roboaudit.core.models import AuditConfig, TemporalWindow, Timeline


@st.composite
def telemetry_strategy(draw):
    """Generate realistic telemetry DataFrames for property testing."""
    num_samples = draw(st.integers(min_value=10, max_value=200))
    fps = draw(st.sampled_from([10.0, 20.0, 30.0]))
    timestamps = np.linspace(0.0, num_samples / fps, num_samples)

    # Gripper states in [0.0, 1.0]
    gripper_states = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=num_samples,
            max_size=num_samples,
        )
    )

    # Velocities >= 0.0
    velocities = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.5),
            min_size=num_samples,
            max_size=num_samples,
        )
    )

    arms = draw(
        st.lists(
            st.sampled_from(["left", "right", "both"]),
            min_size=num_samples,
            max_size=num_samples,
        )
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "gripper_state": gripper_states,
            "velocity": velocities,
            "arm": arms,
        }
    )
    return df


class TestActionPhaseSegmenterProperties:
    """Property-based tests for ActionPhaseSegmenter."""

    @settings(max_examples=100)
    @given(telemetry=telemetry_strategy())
    def test_property_15_temporal_window_classification_completeness(
        self, telemetry: pd.DataFrame
    ):
        """Property 15: Temporal Window Classification Completeness.
        
        Validates Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6.
        All temporal windows must have valid action_phase, arm_attribution,
        contribution_type, valid time bounds, and valid completion_percentage in [0.0, 1.0].
        Advancing phases must be monotonically non-decreasing.
        """
        segmenter = ActionPhaseSegmenter()
        timeline = segmenter.segment_episode(telemetry)

        assert isinstance(timeline, Timeline)
        assert len(timeline.windows) > 0

        valid_phases = {"approach", "grasp", "manipulate", "release", "idle"}
        valid_arms = {"left", "right", "both", "none"}
        valid_contributions = {"advancing", "wasteful", "idle"}

        for w in timeline.windows:
            assert isinstance(w, TemporalWindow)
            assert w.start_s >= 0.0
            assert w.end_s > w.start_s
            assert w.action_phase in valid_phases
            assert w.arm_attribution in valid_arms
            assert w.contribution_type in valid_contributions
            assert 0.0 <= w.completion_percentage <= 1.0

        # Verify monotonicity of advancing windows
        assert timeline.validate_monotonicity() is True

    @settings(max_examples=100)
    @given(
        hesitation_duration=st.floats(min_value=5.1, max_value=20.0),
        phase=st.sampled_from(["approach", "grasp", "manipulate", "release", "idle"]),
    )
    def test_property_17_operator_hesitation_detection(
        self, hesitation_duration: float, phase: str
    ):
        """Property 17: Operator Hesitation Detection.
        
        Validates Requirements: 4.7.
        When an Action_Phase persists > 5.0s without completion progress,
        the segmenter MUST flag an operator_hesitation warning.
        """
        segmenter = ActionPhaseSegmenter()

        # Construct a timeline with a window exceeding 5.0s and 0 progress delta
        hesitating_window = TemporalWindow(
            start_s=2.0,
            end_s=round(2.0 + hesitation_duration, 2),
            action_phase=phase,  # type: ignore[arg-type]
            arm_attribution="right",
            contribution_type="idle" if phase == "idle" else "advancing",
            completion_percentage=0.4,
        )
        timeline = Timeline(windows=[hesitating_window])

        hesitations = segmenter.detect_hesitations(timeline, threshold_s=5.0)

        assert len(hesitations) >= 1
        mistake = hesitations[0]
        assert mistake.type == "hesitation"
        assert mistake.severity in {"low", "medium", "high"}
        assert mistake.t_s == 2.0
        assert mistake.duration_s is not None
        assert mistake.duration_s >= 5.0
        assert len(mistake.evidence) > 0


class TestActionPhaseSegmenterUnit:
    """Unit tests for ActionPhaseSegmenter edge cases and deterministic behaviors."""

    def test_empty_telemetry_returns_empty_timeline(self):
        segmenter = ActionPhaseSegmenter()
        empty_df = pd.DataFrame()
        timeline = segmenter.segment_episode(empty_df)
        assert len(timeline.windows) == 0

    def test_single_row_telemetry_handled_gracefully(self):
        segmenter = ActionPhaseSegmenter()
        df = pd.DataFrame({"timestamp": [1.0], "velocity": [0.0], "gripper_state": [0.0]})
        timeline = segmenter.segment_episode(df)
        assert len(timeline.windows) == 1
        assert timeline.windows[0].start_s == 1.0
        assert timeline.windows[0].end_s > 1.0

    def test_idle_telemetry_generates_idle_phase(self):
        segmenter = ActionPhaseSegmenter()
        # 3 seconds of zero velocity
        timestamps = np.linspace(0.0, 3.0, 30)
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "velocity": np.zeros(30),
                "gripper_state": np.zeros(30),
            }
        )
        timeline = segmenter.segment_episode(df)
        assert len(timeline.windows) >= 1
        assert any(w.action_phase == "idle" for w in timeline.windows)

    def test_grasp_transition_detected(self):
        segmenter = ActionPhaseSegmenter()
        # Gripper opens (0.0) for 1s, then closes (1.0) for 1s
        t = np.linspace(0.0, 2.0, 60)
        gripper = np.concatenate([np.zeros(30), np.ones(30)])
        velocity = np.full(60, 0.2)
        df = pd.DataFrame({"timestamp": t, "gripper_state": gripper, "velocity": velocity})

        timeline = segmenter.segment_episode(df)
        assert len(timeline.windows) >= 1
        phases = [w.action_phase for w in timeline.windows]
        assert "grasp" in phases or "manipulate" in phases

    def test_no_hesitation_when_under_threshold(self):
        segmenter = ActionPhaseSegmenter()
        # Window of 3.0 seconds (below 5.0s threshold)
        window = TemporalWindow(
            start_s=0.0,
            end_s=3.0,
            action_phase="approach",
            arm_attribution="right",
            contribution_type="advancing",
            completion_percentage=0.2,
        )
        timeline = Timeline(windows=[window])
        hesitations = segmenter.detect_hesitations(timeline, threshold_s=5.0)
        assert len(hesitations) == 0
