"""New property-based tests built on the shared Hypothesis strategies module.

These are the net-new properties added for Lesson 4. Each one feeds data from
``roboaudit.tests.strategies`` into a real analyzer and asserts a universal
behavior rather than a single hard-coded example:

- Property A: Clean telemetry (no jitter) never raises a telemetry-jitter issue.
- Property B: An injected sampling gap is always detected as a sensor dropout.
- Property C: Monotonic timelines never trigger a progress-monotonicity violation.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from roboaudit.analysis.dropout import SensorDropoutMonitor
from roboaudit.analysis.invariants import InvariantChecker
from roboaudit.analysis.timebase import TimebaseVerifier
from roboaudit.core.models import AuditConfig, TaskOutcome
from roboaudit.tests import strategies as rst


class TestNewLesson4Properties:
    """Property-based tests exercising analyzers with shared generated data."""

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(telemetry=rst.telemetry_streams(jitter_fraction=0.0, rate_hz=30.0))
    def test_property_a_clean_telemetry_has_no_jitter_issue(self, telemetry):
        """Property A: perfectly regular telemetry (zero jitter) must not be
        flagged with a telemetry_jitter issue at the 2% default threshold."""
        verifier = TimebaseVerifier(AuditConfig())
        issues, stats = verifier.check_telemetry_timebase(telemetry, expected_rate_hz=30.0)
        assert not any(i.issue_type == "telemetry_jitter" for i in issues)
        assert stats.sample_count == len(telemetry)

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        gap_index=st.integers(min_value=2, max_value=6),
        gap_size_s=st.floats(min_value=0.25, max_value=1.0),
    )
    def test_property_b_injected_gap_is_detected_as_dropout(self, gap_index, gap_size_s):
        """Property B: a telemetry stream with a gap larger than the 100ms
        threshold always yields at least one sensor_dropout issue, and the
        reported gap duration is at least the injected gap."""
        # Build a perfectly-regular 30Hz stream, then inject one gap after
        # ``gap_index``. (Done directly rather than via a strategy, since a
        # Hypothesis strategy must not be sampled inside a @given body.)
        n = gap_index + 6
        interval = 1.0 / 30.0
        timestamps = []
        t = 0.0
        for i in range(n):
            t += interval
            if i == gap_index:
                t += gap_size_s
            timestamps.append(round(t, 6))
        telemetry = pd.DataFrame(
            {"timestamp": timestamps, "gripper_pos": [round(0.01 * i, 6) for i in range(n)]}
        )

        monitor = SensorDropoutMonitor(AuditConfig())
        dropouts = monitor.detect_dropouts(telemetry)
        assert len(dropouts) >= 1
        assert all(d.issue_type == "sensor_dropout" for d in dropouts)
        assert max(d.duration_s for d in dropouts) >= gap_size_s - 0.05

    @settings(max_examples=100)
    @given(timeline=rst.timelines(monotonic=True))
    def test_property_c_monotonic_timeline_no_violation(self, timeline):
        """Property C: a non-decreasing (monotonic) timeline must never produce
        a progress_monotonicity invariant violation."""
        checker = InvariantChecker()
        violations = checker.check_progress_monotonicity(timeline)
        assert not any(v.invariant_name == "progress_monotonicity" for v in violations)
