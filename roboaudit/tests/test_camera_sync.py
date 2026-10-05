"""Tests for CameraSyncChecker and multi-camera consistency.

Covers:
- Property 41: Multi-Camera Timestamp Synchronization (Req 10.1, 10.2)
- Unit tests for frame count mismatch (Req 10.3)
- Unit tests for camera stream swap detection (Req 10.4, 10.5)
- Unit tests for DataIssue conversion and clean multi-camera streams
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roboaudit.analysis.camera_sync import CameraSyncChecker, CameraSyncIssue
from roboaudit.core.models import AuditConfig, Frame


@st.composite
def desynced_camera_streams_strategy(draw):
    """Generate two camera streams where timestamps intentionally diverge beyond 16ms."""
    num_frames = draw(st.integers(min_value=15, max_value=60))
    fps = 30.0
    interval_us = int(1_000_000 / fps)

    # Frame timestamps for cam_a
    timestamps_a = [i * interval_us for i in range(num_frames)]

    # Lag beyond 16ms (e.g. 17ms to 100ms = 17,000 to 100,000 µs)
    lag_us = draw(st.integers(min_value=17_000, max_value=100_000))
    timestamps_b = [t + lag_us for t in timestamps_a]

    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)

    frames_a = [
        Frame(timestamp_us=ts, pts=i, image=dummy_img, camera_id="cam_front")
        for i, ts in enumerate(timestamps_a)
    ]
    frames_b = [
        Frame(timestamp_us=ts, pts=i, image=dummy_img, camera_id="cam_wrist")
        for i, ts in enumerate(timestamps_b)
    ]

    return {"cam_front": frames_a, "cam_wrist": frames_b}, lag_us / 1000.0


class TestCameraSyncCheckerProperties:
    """Property-based tests for CameraSyncChecker."""

    @settings(max_examples=100)
    @given(data=desynced_camera_streams_strategy())
    def test_property_41_multi_camera_timestamp_synchronization(self, data):
        """Property 41: Multi-Camera Timestamp Synchronization.
        
        Validates Requirements: 10.1, 10.2.
        WHEN camera frame timestamps diverge by more than 16 milliseconds,
        THE Audit_Engine SHALL flag a camera_desync warning with the affected camera pair.
        """
        streams, expected_lag_ms = data
        checker = CameraSyncChecker()
        issues = checker.check_timestamp_sync(streams, tolerance_ms=16.0)

        assert len(issues) >= 1
        issue = issues[0]
        assert issue.issue_type == "desync"
        assert issue.severity == "warning"
        assert issue.camera_pair == ("cam_front", "cam_wrist")
        assert issue.time_range is not None


class TestCameraSyncCheckerUnit:
    """Unit tests for CameraSyncChecker."""

    def test_synchronized_cameras_have_zero_issues(self):
        """Test cameras within 16ms tolerance report no desync issues."""
        frames_a = [
            Frame(timestamp_us=i * 33333, pts=i, image=np.zeros((10, 10, 3), dtype=np.uint8), camera_id="front")
            for i in range(30)
        ]
        # Introduce tiny jitter well below 16ms (e.g. 5ms = 5000us)
        frames_b = [
            Frame(timestamp_us=i * 33333 + 5000, pts=i, image=np.zeros((10, 10, 3), dtype=np.uint8), camera_id="wrist")
            for i in range(30)
        ]

        checker = CameraSyncChecker()
        issues = checker.check_timestamp_sync({"front": frames_a, "wrist": frames_b}, tolerance_ms=16.0)
        assert len(issues) == 0

    def test_frame_count_mismatch_detected(self):
        """Test Requirement 10.3: counts differing by > 2 frames flagged."""
        frames_a = [
            Frame(timestamp_us=i * 33333, pts=i, image=np.zeros((10, 10, 3), dtype=np.uint8), camera_id="front")
            for i in range(30)
        ]
        # Wrist has 25 frames (diff of 5 > tolerance of 2)
        frames_b = [
            Frame(timestamp_us=i * 33333, pts=i, image=np.zeros((10, 10, 3), dtype=np.uint8), camera_id="wrist")
            for i in range(25)
        ]

        checker = CameraSyncChecker()
        issues = checker.check_frame_counts({"front": frames_a, "wrist": frames_b}, tolerance_frames=2)

        assert len(issues) == 1
        issue = issues[0]
        assert issue.issue_type == "frame_count_mismatch"
        assert issue.severity == "warning"
        assert issue.camera_pair == ("front", "wrist")

    def test_frame_count_within_tolerance_accepted(self):
        """Difference of 1 frame is within tolerance (<= 2)."""
        frames_a = [
            Frame(timestamp_us=i * 33333, pts=i, image=np.zeros((10, 10, 3), dtype=np.uint8), camera_id="front")
            for i in range(30)
        ]
        frames_b = [
            Frame(timestamp_us=i * 33333, pts=i, image=np.zeros((10, 10, 3), dtype=np.uint8), camera_id="wrist")
            for i in range(29)
        ]

        checker = CameraSyncChecker()
        issues = checker.check_frame_counts({"front": frames_a, "wrist": frames_b}, tolerance_frames=2)
        assert len(issues) == 0

    def test_camera_swap_detected(self):
        """Test Requirements 10.4, 10.5: High severity camera_swap error detected."""
        # Camera A is predominantly blue (0, 0, 200), Camera B is red (200, 0, 0)
        blue_img = np.full((16, 16, 3), (0, 0, 200), dtype=np.uint8)
        red_img = np.full((16, 16, 3), (200, 0, 0), dtype=np.uint8)

        # In first 10 frames: front=blue, wrist=red
        # In last 10 frames: front=red, wrist=blue (swapped!)
        frames_a = []
        frames_b = []

        for i in range(10):
            t_us = i * 33333
            frames_a.append(Frame(timestamp_us=t_us, pts=i, image=blue_img, camera_id="front"))
            frames_b.append(Frame(timestamp_us=t_us, pts=i, image=red_img, camera_id="wrist"))

        for i in range(10, 20):
            t_us = i * 33333
            frames_a.append(Frame(timestamp_us=t_us, pts=i, image=red_img, camera_id="front"))
            frames_b.append(Frame(timestamp_us=t_us, pts=i, image=blue_img, camera_id="wrist"))

        checker = CameraSyncChecker()
        issues = checker.detect_camera_swap({"front": frames_a, "wrist": frames_b})

        assert len(issues) >= 1
        swap_issue = issues[0]
        assert swap_issue.issue_type == "camera_swap"
        assert swap_issue.severity == "error"
        assert swap_issue.camera_pair == ("front", "wrist")

    def test_to_data_issue_conversion(self):
        issue = CameraSyncIssue(
            issue_type="desync",
            severity="warning",
            camera_pair=("cam_1", "cam_2"),
            time_range=(1.2, 3.4),
            details="Timestamp lag",
        )
        data_issue = issue.to_data_issue()
        assert data_issue.category == "camera"
        assert data_issue.severity == "medium"
        assert data_issue.t_s == 1.2
        assert "cam_1_vs_cam_2" in data_issue.evidence
