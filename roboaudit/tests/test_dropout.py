"""Tests for SensorDropoutMonitor, freeze detection, and outlier analysis.

Covers:
- Property 44: Sensor Dropout Detection (Req 11.1, 11.5)
- Property 45: Sensor Freeze Detection (Req 11.2)
- Unit tests for sensor outlier detection and DataIssue conversion
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roboaudit.analysis.dropout import DropoutIssue, SensorDropoutMonitor
from roboaudit.core.models import AuditConfig


@st.composite
def dropout_telemetry_strategy(draw):
    """Generate telemetry with an injected gap > 100ms."""
    num_samples = draw(st.integers(min_value=20, max_value=80))
    dt_nominal = 0.033  # ~30Hz
    gap_s = draw(st.floats(min_value=0.11, max_value=2.0))  # > 100ms
    gap_index = draw(st.integers(min_value=5, max_value=num_samples - 5))

    timestamps = []
    current_t = 0.0
    for i in range(num_samples):
        if i == gap_index:
            current_t += gap_s
        else:
            current_t += dt_nominal
        timestamps.append(current_t)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "joint_1": np.sin(np.linspace(0, 5, num_samples)),
        }
    )
    t_start = timestamps[gap_index - 1]
    t_end = timestamps[gap_index]
    return df, t_start, t_end


@st.composite
def frozen_telemetry_strategy(draw):
    """Generate telemetry where one channel freezes for > 1.0s while others move."""
    num_samples = 60
    fps = 30.0
    timestamps = np.linspace(0.0, 2.0, num_samples)  # 2.0s duration

    # Joint 1 is active throughout
    joint_1 = np.linspace(0.0, 1.5, num_samples)

    # Joint 2 is frozen for > 1.0s (e.g. 45 samples = 1.5s)
    freeze_val = draw(st.floats(min_value=-2.0, max_value=2.0))
    joint_2 = np.full(num_samples, freeze_val)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "joint_1": joint_1,
            "joint_2": joint_2,
        }
    )
    return df


class TestSensorDropoutMonitorProperties:
    """Property-based tests for SensorDropoutMonitor."""

    @settings(max_examples=100)
    @given(data=dropout_telemetry_strategy())
    def test_property_44_sensor_dropout_detection(self, data):
        """Property 44: Sensor Dropout Detection.
        
        Validates Requirements: 11.1, 11.5.
        WHEN telemetry channels exhibit gaps exceeding 100 milliseconds,
        THE Audit_Engine SHALL flag a sensor_dropout warning with the affected
        channel, time range, duration, and percentage.
        """
        df, t_start, t_end = data
        monitor = SensorDropoutMonitor()
        issues = monitor.detect_dropouts(df, gap_threshold_ms=100.0)

        assert len(issues) >= 1
        issue = issues[0]
        assert issue.issue_type == "sensor_dropout"
        assert issue.severity == "warning"
        assert issue.duration_s >= 0.10
        assert issue.percentage_of_episode > 0.0
        assert abs(issue.time_range[0] - t_start) < 0.05
        assert abs(issue.time_range[1] - t_end) < 0.05

    @settings(max_examples=100)
    @given(df=frozen_telemetry_strategy())
    def test_property_45_sensor_freeze_detection(self, df):
        """Property 45: Sensor Freeze Detection.
        
        Validates Requirements: 11.2.
        WHEN telemetry values remain constant for more than 1.0 second during motion,
        THE Audit_Engine SHALL flag a sensor_freeze warning.
        """
        monitor = SensorDropoutMonitor()
        issues = monitor.detect_frozen_sensors(df, freeze_duration_s=1.0)

        assert len(issues) >= 1
        freeze_issue = next(i for i in issues if i.channel == "joint_2")
        assert freeze_issue.issue_type == "sensor_freeze"
        assert freeze_issue.severity == "warning"
        assert freeze_issue.duration_s >= 1.0


class TestSensorDropoutMonitorUnit:
    """Unit tests for SensorDropoutMonitor."""

    def test_clean_telemetry_zero_dropouts(self):
        """Clean 30Hz stream has zero dropout issues."""
        timestamps = np.linspace(0.0, 1.0, 30)
        df = pd.DataFrame({"timestamp": timestamps, "joint_0": np.zeros(30)})
        monitor = SensorDropoutMonitor()
        issues = monitor.detect_dropouts(df, gap_threshold_ms=100.0)
        assert len(issues) == 0

    def test_sensor_outlier_detection(self):
        """Test Requirements 11.3, 11.4: outlier values exceeding joint limits flagged."""
        # Joint limit is [-3.14, 3.14]; insert an outlier value of 15.0 rad
        df = pd.DataFrame(
            {
                "timestamp": [0.0, 0.1, 0.2],
                "joint_elbow": [0.5, 15.0, 0.6],
            }
        )
        monitor = SensorDropoutMonitor()
        issues = monitor.detect_outliers(
            df, joint_limits={"joint_elbow": (-3.14, 3.14)}
        )

        assert len(issues) == 1
        issue = issues[0]
        assert issue.issue_type == "sensor_outlier"
        assert issue.severity == "warning"
        assert issue.channel == "joint_elbow"

    def test_to_data_issue_conversion(self):
        issue = DropoutIssue(
            issue_type="sensor_dropout",
            severity="warning",
            channel="joint_shoulder",
            time_range=(2.0, 2.5),
            duration_s=0.5,
            percentage_of_episode=10.0,
            details="500ms bus dropout",
        )
        data_issue = issue.to_data_issue()
        assert data_issue.category == "sensor"
        assert data_issue.severity == "medium"
        assert data_issue.t_s == 2.0
        assert "joint_shoulder" in data_issue.evidence[0]
