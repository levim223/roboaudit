"""Tests for GraspAnomalyDetector and optical flow elevation measurement.

Covers:
- Property 11: Phantom Grasp Detection (Req 3.1)
- Property 13: Object Drop Detection (Req 3.3)
- Property 14: Anomaly Evidence Completeness (Req 3.4)
- Unit tests for sensor mismatch, optical flow elevation measurement, and DataIssue conversion
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roboaudit.analysis.grasp import GraspAnomaly, GraspAnomalyDetector
from roboaudit.core.models import AuditConfig, Frame


@st.composite
def phantom_grasp_telemetry_strategy(draw):
    """Generate telemetry with closed gripper and elevation delta == 0."""
    t_val = draw(st.floats(min_value=0.5, max_value=60.0))
    grip_val = draw(st.floats(min_value=0.75, max_value=1.0))  # Closed gripper

    df = pd.DataFrame(
        {
            "timestamp": [round(t_val, 2)],
            "gripper_state": [grip_val],
            "elevation_delta": [0.0],
        }
    )
    return df, t_val


@st.composite
def missed_drop_telemetry_strategy(draw):
    """Generate telemetry with closed gripper and visual object falling == True."""
    t_val = draw(st.floats(min_value=0.5, max_value=60.0))
    grip_val = draw(st.floats(min_value=0.75, max_value=1.0))  # Closed gripper

    df = pd.DataFrame(
        {
            "timestamp": [round(t_val, 2)],
            "gripper_state": [grip_val],
            "object_falling": [True],
        }
    )
    return df, t_val


class TestGraspAnomalyDetectorProperties:
    """Property-based tests for GraspAnomalyDetector."""

    @settings(max_examples=100)
    @given(data=phantom_grasp_telemetry_strategy())
    def test_property_11_phantom_grasp_detection(self, data):
        """Property 11: Phantom Grasp Detection.
        
        Validates Requirements: 3.1.
        WHEN gripper motor registers closed AND visual object elevation delta is zero,
        THE Audit_Engine SHALL record a grasp_anomaly with high severity.
        """
        telemetry, t_val = data
        detector = GraspAnomalyDetector()
        anomalies = detector.detect_anomalies(telemetry)

        assert len(anomalies) == 1
        anomaly = anomalies[0]
        assert anomaly.anomaly_type == "phantom_grasp"
        assert anomaly.severity == "high"
        assert abs(anomaly.timestamp_s - t_val) < 0.05
        assert len(anomaly.visual_evidence) > 0

    @settings(max_examples=100)
    @given(data=missed_drop_telemetry_strategy())
    def test_property_13_object_drop_detection(self, data):
        """Property 13: Object Drop Detection.
        
        Validates Requirements: 3.3.
        WHEN visual evidence shows object dropping AND gripper state remains closed,
        THE Audit_Engine SHALL record a grasp_anomaly with high severity.
        """
        telemetry, t_val = data
        detector = GraspAnomalyDetector()
        anomalies = detector.detect_anomalies(telemetry)

        assert len(anomalies) == 1
        anomaly = anomalies[0]
        assert anomaly.anomaly_type == "missed_drop"
        assert anomaly.severity == "high"
        assert abs(anomaly.timestamp_s - t_val) < 0.05
        assert len(anomaly.visual_evidence) > 0

    @settings(max_examples=100)
    @given(
        is_phantom=st.booleans(),
        t_s=st.floats(min_value=0.0, max_value=100.0),
        grip_val=st.floats(min_value=0.75, max_value=1.0),
    )
    def test_property_14_anomaly_evidence_completeness(
        self, is_phantom: bool, t_s: float, grip_val: float
    ):
        """Property 14: Anomaly Evidence Completeness.
        
        Validates Requirements: 3.4.
        FOR ALL grasp_anomaly detections, THE Audit_Engine SHALL include
        Camera_Evidence references with frame numbers.
        """
        df = pd.DataFrame(
            {
                "timestamp": [round(t_s, 2)],
                "gripper_state": [grip_val],
                "elevation_delta": [0.0 if is_phantom else 0.15],
                "object_falling": [False if is_phantom else True],
            }
        )
        detector = GraspAnomalyDetector()
        anomalies = detector.detect_anomalies(df)

        for anomaly in anomalies:
            assert len(anomaly.visual_evidence) > 0
            for ref in anomaly.visual_evidence:
                assert "frame_" in ref or "pts" in ref or ":" in ref


class TestGraspAnomalyDetectorUnit:
    """Unit tests for GraspAnomalyDetector edge cases and helper methods."""

    def test_sensor_mismatch_detection(self):
        """Test Requirement 3.2: Force > 0 and no object in bbox -> medium severity."""
        df = pd.DataFrame(
            {
                "timestamp": [1.5],
                "gripper_state": [0.3],
                "gripper_force": [5.2],  # > 0.5N
                "object_in_gripper": [False],
            }
        )
        detector = GraspAnomalyDetector()
        anomalies = detector.detect_anomalies(df)

        assert len(anomalies) == 1
        anomaly = anomalies[0]
        assert anomaly.anomaly_type == "sensor_mismatch"
        assert anomaly.severity == "medium"
        assert anomaly.timestamp_s == 1.5

    def test_nominal_grasp_no_anomalies(self):
        """Test that normal successful grasp (closed + elevation delta > 0) has 0 anomalies."""
        df = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "gripper_state": [0.9, 0.9],
                "elevation_delta": [0.08, 0.12],  # Object lifted
                "object_falling": [False, False],
                "gripper_force": [2.0, 2.0],
                "object_in_gripper": [True, True],
            }
        )
        detector = GraspAnomalyDetector()
        anomalies = detector.detect_anomalies(df)
        assert len(anomalies) == 0

    def test_to_data_issue_conversion(self):
        anomaly = GraspAnomaly(
            anomaly_type="phantom_grasp",
            severity="high",
            timestamp_s=2.5,
            gripper_state="closed",
            visual_evidence=["frame_75"],
            details="Phantom grasp verified",
        )
        issue = anomaly.to_data_issue()
        assert issue.category == "grasp"
        assert issue.severity == "high"
        assert issue.t_s == 2.5
        assert "frame_75" in issue.evidence

    def test_measure_object_elevation_optical_flow(self):
        """Test Requirement 3.5: measure_object_elevation using frames."""
        # Create two simple 64x64 synthetic frames with simulated upward motion
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.zeros((64, 64, 3), dtype=np.uint8)

        # Place a bright square at y=40 in frame 1, and y=30 (moved up) in frame 2
        img1[35:45, 25:35] = 255
        img2[25:35, 25:35] = 255

        frame1 = Frame(timestamp_us=0, pts=0, image=img1, camera_id="front")
        frame2 = Frame(timestamp_us=33333, pts=1, image=img2, camera_id="front")

        detector = GraspAnomalyDetector()
        delta = detector.measure_object_elevation([frame1, frame2])

        # Optical flow should indicate upward motion (positive elevation delta)
        assert delta > 0.0
