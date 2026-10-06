"""Shared Hypothesis strategies for RoboAudit property-based tests (Lesson 4 / task 16.1).

Centralizes generation of valid core domain objects so property tests across the
suite can share one source of truth instead of hand-rolling data inline. Every
strategy is guaranteed to produce objects that satisfy the model invariants
defined in ``roboaudit.core.models`` (e.g. ``TemporalWindow`` requires
``end_s > start_s`` and ``0.0 <= completion_percentage <= 1.0``).

Public strategies:
    temporal_windows()   -> TemporalWindow
    timelines()          -> Timeline (chronological, monotonic completion)
    telemetry_streams()  -> pandas.DataFrame with controllable jitter/gaps
    episodes()           -> ExtractedEpisode
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from hypothesis import strategies as st

from roboaudit.core.models import (
    EpisodeMetadata,
    ExtractedEpisode,
    TemporalWindow,
    Timeline,
)

ACTION_PHASES = ["approach", "grasp", "manipulate", "release", "idle"]
ARMS = ["left", "right", "both", "none"]
CONTRIBUTIONS = ["advancing", "wasteful", "idle"]
OUTCOMES = ["success", "failure", "partial", "success_then_undone"]


@st.composite
def temporal_windows(
    draw,
    start_s: float = 0.0,
    min_duration: float = 0.5,
    max_duration: float = 5.0,
    completion: float | None = None,
) -> TemporalWindow:
    """Generate a single valid TemporalWindow starting at ``start_s``.

    The ``end_s > start_s`` invariant is guaranteed even after 2-decimal rounding.
    """
    duration = draw(st.floats(min_value=min_duration, max_value=max_duration))
    start_r = round(start_s, 2)
    end_r = round(start_s + duration, 2)
    if end_r <= start_r:
        end_r = round(start_r + max(min_duration, 0.5), 2)

    pct = completion if completion is not None else draw(
        st.floats(min_value=0.0, max_value=1.0)
    )
    return TemporalWindow(
        start_s=start_r,
        end_s=end_r,
        action_phase=draw(st.sampled_from(ACTION_PHASES)),
        arm_attribution=draw(st.sampled_from(ARMS)),
        contribution_type=draw(st.sampled_from(CONTRIBUTIONS)),
        completion_percentage=round(pct, 2),
    )


@st.composite
def timelines(
    draw,
    min_windows: int = 1,
    max_windows: int = 6,
    monotonic: bool = True,
) -> Timeline:
    """Generate a chronologically-ordered Timeline.

    When ``monotonic`` is True (default), completion is non-decreasing across the
    whole timeline, which guarantees ``Timeline.validate_monotonicity()`` holds
    regardless of contribution types.
    """
    n = draw(st.integers(min_value=min_windows, max_value=max_windows))
    windows: List[TemporalWindow] = []
    t = 0.0
    prev_completion = 0.0
    for i in range(n):
        if monotonic:
            # Non-decreasing completion, ending at ~1.0 for the last window.
            step = (1.0 - prev_completion) * draw(st.floats(min_value=0.0, max_value=1.0))
            completion = min(1.0, prev_completion + step)
            prev_completion = completion
        else:
            completion = None  # free-form
        w = draw(temporal_windows(start_s=t, completion=completion))
        windows.append(w)
        t = w.end_s
    return Timeline(windows=windows)


@st.composite
def telemetry_streams(
    draw,
    channels: tuple[str, ...] = ("gripper_pos", "joint_1", "ee_z"),
    min_samples: int = 10,
    max_samples: int = 60,
    rate_hz: float = 30.0,
    jitter_fraction: float = 0.0,
    gap_at: int | None = None,
    gap_s: float = 0.0,
) -> pd.DataFrame:
    """Generate a telemetry DataFrame with a ``timestamp`` column (seconds).

    Args:
        jitter_fraction: Fraction of the nominal interval to randomly perturb each
            sample timestamp by (0.0 = perfectly regular sampling).
        gap_at: Optional sample index after which to insert a dropout gap.
        gap_s: Size of the inserted gap in seconds.
    """
    n = draw(st.integers(min_value=min_samples, max_value=max_samples))
    interval = 1.0 / rate_hz
    timestamps = []
    t = 0.0
    for i in range(n):
        if jitter_fraction > 0.0:
            delta = draw(
                st.floats(
                    min_value=interval * (1.0 - jitter_fraction),
                    max_value=interval * (1.0 + jitter_fraction),
                )
            )
        else:
            delta = interval
        t += delta
        if gap_at is not None and i == gap_at:
            t += gap_s
        timestamps.append(round(t, 6))

    data = {"timestamp": timestamps}
    for ch in channels:
        base = draw(st.floats(min_value=-1.0, max_value=1.0))
        data[ch] = [round(base + 0.01 * i, 6) for i in range(n)]
    return pd.DataFrame(data)


@st.composite
def episode_metadata(draw) -> EpisodeMetadata:
    """Generate valid EpisodeMetadata."""
    outcome = draw(st.sampled_from(OUTCOMES))
    return EpisodeMetadata(
        episode_id=f"ep_{draw(st.integers(min_value=1, max_value=99999))}",
        duration_s=draw(st.floats(min_value=5.0, max_value=120.0)),
        recording_date=datetime(2026, 9, 21, tzinfo=timezone.utc),
        robot_type=draw(st.sampled_from(["umi", "franka", "kinova", "generic"])),
        camera_count=draw(st.integers(min_value=1, max_value=3)),
        telemetry_channels=["gripper_pos", "joint_1"],
        task_outcome=outcome,
    )


@st.composite
def episodes(draw) -> ExtractedEpisode:
    """Generate a complete ExtractedEpisode (metadata + telemetry, no video frames).

    Video frames are left empty because generating decodable image arrays adds
    cost without exercising more logic for most property tests.
    """
    meta = draw(episode_metadata())
    telemetry = draw(telemetry_streams())
    return ExtractedEpisode(
        metadata=meta,
        video_streams={},
        telemetry=telemetry,
    )
