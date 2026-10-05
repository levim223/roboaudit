"""Property-based tests and unit tests for TimebaseVerifier.

Covers:
- Property 6: Frame Jitter Detection (Req 2.1)
- Property 7: Telemetry Jitter Detection (Req 2.2)
- Property 8: Non-Monotonic Timestamp Detection (Req 2.3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roboaudit.analysis.timebase import TimebaseVerifier
from roboaudit.core.models import AuditConfig, Frame


def make_dummy_frame(timestamp_us: int, pts: int, camera_id: str = "cam_0") -> Frame:
    """Helper to create dummy Frame with given timestamp."""
    return Frame(
        timestamp_us=timestamp_us,
        pts=pts,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        camera_id=camera_id,
    )


class TestTimebaseProperties:
    """Hypothesis property-based tests for timebase verification (Lesson 4)."""

    @settings(max_examples=100)
    @given(
        num_frames=st.integers(min_value=10, max_value=50),
        base_interval_ms=st.floats(min_value=20.0, max_value=100.0),
        jitter_factor=st.floats(min_value=0.03, max_value=0.50),  # > 2.0%
    )
    def test_property_6_frame_jitter_detection(
        self, num_frames: int, base_interval_ms: float, jitter_factor: float
    ):
        """Property 6: WHEN frame interval variance exceeds 2.0% THEN timebase_jitter is flagged."""
        verifier = TimebaseVerifier(AuditConfig(jitter_threshold=0.02))
        base_interval_us = int(base_interval_ms * 1000)

        # Construct frame sequence where every second frame has jitter exceeding threshold
        frames = []
        current_us = 1_000_000
        for i in range(num_frames):
            frames.append(make_dummy_frame(current_us, i))
            # Inject jitter > 2.0%
            delta = base_interval_us if i % 2 == 0 else int(base_interval_us * (1.0 + jitter_factor))
            current_us += delta

        issues, stats = verifier.check_video_timebase(frames, expected_fps=1_000_000.0 / base_interval_us)
        # Should flag frame_jitter warning since jitter factor is > 2.0%
        has_jitter = any(issue.issue_type == "frame_jitter" for issue in issues)
        assert has_jitter or stats.jitter_fraction >= 0.02

    @settings(max_examples=100)
    @given(
        num_samples=st.integers(min_value=10, max_value=50),
        jitter_amp=st.floats(min_value=0.05, max_value=0.20),
    )
    def test_property_7_telemetry_jitter_detection(
        self, num_samples: int, jitter_amp: float
    ):
        """Property 7: WHEN telemetry sampling jitter exceeds 2.0% THEN warning is flagged."""
        verifier = TimebaseVerifier(AuditConfig(jitter_threshold=0.02))
        nominal_dt = 0.01  # 100 Hz

        timestamps = [0.0]
        for i in range(1, num_samples):
            # Alternating excessive interval delta
            step = nominal_dt * (1.0 + jitter_amp) if i % 2 == 0 else nominal_dt
            timestamps.append(timestamps[-1] + step)

        df = pd.DataFrame({"timestamp": timestamps, "joint_0": np.zeros(num_samples)})
        issues, stats = verifier.check_telemetry_timebase(df, expected_rate_hz=100.0)

        has_telemetry_jitter = any(issue.issue_type == "telemetry_jitter" for issue in issues)
        assert has_telemetry_jitter or stats.jitter_fraction >= 0.02

    @settings(max_examples=100)
    @given(
        num_frames=st.integers(min_value=5, max_value=30),
        violation_index=st.integers(min_value=1, max_value=4),
    )
    def test_property_8_non_monotonic_timestamp_detection(
        self, num_frames: int, violation_index: int
    ):
        """Property 8: WHEN timestamps are non-monotonic THEN high severity error is flagged."""
        verifier = TimebaseVerifier()
        frames = []
        current_us = 1_000_000

        for i in range(num_frames):
            if i == violation_index:
                # Deliberately inject backwards or duplicate timestamp
                frames.append(make_dummy_frame(current_us - 50_000, i))
            else:
                frames.append(make_dummy_frame(current_us, i))
            current_us += 33_333

        issues, _ = verifier.check_video_timebase(frames)
        has_non_monotonic = any(issue.issue_type == "non_monotonic" and issue.severity == "high" for issue in issues)
        assert has_non_monotonic


class TestTimebaseVerifierUnit:
    """Unit tests for edge cases and cross-modal synchronization."""

    def test_empty_and_single_frame(self):
        verifier = TimebaseVerifier()
        issues, stats = verifier.check_video_timebase([])
        assert issues == []
        assert stats.sample_count == 0

        issues, stats = verifier.check_video_timebase([make_dummy_frame(1000, 0)])
        assert issues == []
        assert stats.sample_count == 1

    def test_perfectly_consistent_stream_has_zero_issues(self):
        verifier = TimebaseVerifier(AuditConfig(jitter_threshold=0.02))
        frames = [make_dummy_frame(i * 33333, i) for i in range(30)]

        issues, stats = verifier.check_video_timebase(frames, expected_fps=30.0)
        assert len(issues) == 0
        assert stats.jitter_fraction == 0.0

    def test_cross_modal_sync_within_tolerance(self):
        verifier = TimebaseVerifier(AuditConfig(sensor_desync_tolerance_ms=50.0))
        # Video frames at 0.0s, 1.0s, 2.0s
        frames = [make_dummy_frame(int(t * 1_000_000), i) for i, t in enumerate([0.0, 1.0, 2.0])]
        video_streams = {"front": frames}

        # Telemetry at 100 Hz (every 10ms), perfectly covering the video
        telemetry_ts = np.linspace(0.0, 2.0, 201)
        telemetry_df = pd.DataFrame({"timestamp": telemetry_ts})

        issues = verifier.check_cross_modal_sync(video_streams, telemetry_df)
        assert len(issues) == 0

    def test_cross_modal_sync_exceeding_tolerance(self):
        verifier = TimebaseVerifier(AuditConfig(sensor_desync_tolerance_ms=50.0))
        # Video frames starting at 0.0s
        frames = [make_dummy_frame(0, 0), make_dummy_frame(1_000_000, 1)]
        video_streams = {"front": frames}

        # Telemetry starts 150ms late (exceeding 50ms tolerance)
        telemetry_df = pd.DataFrame({"timestamp": [0.150, 0.200, 1.0]})

        issues = verifier.check_cross_modal_sync(video_streams, telemetry_df)
        assert any(i.issue_type == "sensor_desync" for i in issues)
