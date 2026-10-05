"""Timebase and cross-modal synchronization verifier for robotics demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from roboaudit.core.models import AuditConfig, Frame


@dataclass
class TimebaseIssue:
    """Represents a temporal anomaly detected in video or telemetry streams."""

    issue_type: Literal[
        "frame_jitter",
        "telemetry_jitter",
        "non_monotonic",
        "sensor_desync",
        "sped_up_playback",
    ]
    severity: Literal["low", "medium", "high", "error"]
    time_range: Tuple[float, float]
    affected_stream: str
    details: str


@dataclass
class TimebaseStats:
    """Statistical summary of sampling intervals for a temporal stream."""

    mean_interval_s: float
    std_interval_s: float
    jitter_fraction: float
    sample_count: int


class TimebaseVerifier:
    """Verifies timing consistency, jitter thresholds, and cross-modal sync."""

    def __init__(self, config: Optional[AuditConfig] = None) -> None:
        """Initialize TimebaseVerifier with configuration thresholds."""
        self.config = config or AuditConfig()

    def check_video_timebase(
        self,
        frames: List[Frame],
        camera_id: str = "camera_0",
        expected_fps: Optional[float] = None,
    ) -> Tuple[List[TimebaseIssue], TimebaseStats]:
        """Verify video frame timing consistency, jitter, and monotonicity.
        
        Args:
            frames: Chronological list of Frame objects.
            camera_id: Identifier of the camera stream.
            expected_fps: Optional nominal FPS. If omitted, median delta is used.
            
        Returns:
            Tuple of (List[TimebaseIssue], TimebaseStats).
        """
        if len(frames) < 2:
            return (
                [],
                TimebaseStats(
                    mean_interval_s=0.0,
                    std_interval_s=0.0,
                    jitter_fraction=0.0,
                    sample_count=len(frames),
                ),
            )

        timestamps_s = np.array([f.timestamp_us / 1_000_000.0 for f in frames], dtype=np.float64)
        issues: List[TimebaseIssue] = []

        # 1. Check for non-monotonic timestamps
        deltas = np.diff(timestamps_s)
        non_mono_indices = np.where(deltas <= 0)[0]
        for idx in non_mono_indices:
            t_curr = timestamps_s[idx]
            t_next = timestamps_s[idx + 1]
            issues.append(
                TimebaseIssue(
                    issue_type="non_monotonic",
                    severity="high",
                    time_range=(float(t_curr), float(t_next)),
                    affected_stream=camera_id,
                    details=(
                        f"Non-monotonic frame timestamp detected between frame {idx} ({t_curr:.4f}s) "
                        f"and frame {idx + 1} ({t_next:.4f}s)"
                    ),
                )
            )

        # 2. Compute expected interval
        if expected_fps is not None and expected_fps > 0:
            expected_interval = 1.0 / expected_fps
        else:
            positive_deltas = deltas[deltas > 0]
            expected_interval = float(np.median(positive_deltas)) if len(positive_deltas) > 0 else 1.0 / 30.0

        # 3. Calculate jitter variance: abs(delta_t - expected_interval) / expected_interval >= threshold
        if expected_interval > 0:
            jitter_mask = np.abs(deltas - expected_interval) / expected_interval >= self.config.jitter_threshold
            jitter_indices = np.where(jitter_mask)[0]
            jitter_fraction = float(np.mean(jitter_mask)) if len(deltas) > 0 else 0.0

            if jitter_fraction >= self.config.jitter_threshold and len(jitter_indices) > 0:
                t_start = float(timestamps_s[jitter_indices[0]])
                t_end = float(timestamps_s[jitter_indices[-1] + 1])
                issues.append(
                    TimebaseIssue(
                        issue_type="frame_jitter",
                        severity="medium",
                        time_range=(t_start, t_end),
                        affected_stream=camera_id,
                        details=(
                            f"Frame jitter fraction ({jitter_fraction:.2%}) exceeds threshold "
                            f"({self.config.jitter_threshold:.2%}) with expected interval {expected_interval:.4f}s"
                        ),
                    )
                )
        else:
            jitter_fraction = 0.0

        stats = TimebaseStats(
            mean_interval_s=float(np.mean(deltas)),
            std_interval_s=float(np.std(deltas)),
            jitter_fraction=jitter_fraction,
            sample_count=len(frames),
        )

        return issues, stats

    def check_telemetry_timebase(
        self,
        telemetry: pd.DataFrame,
        channel: Optional[str] = None,
        expected_rate_hz: Optional[float] = None,
    ) -> Tuple[List[TimebaseIssue], TimebaseStats]:
        """Verify telemetry timestamp consistency and jitter.
        
        Args:
            telemetry: Telemetry DataFrame with 'timestamp' column.
            channel: Identifier of the sensor channel.
            expected_rate_hz: Nominal sensor frequency in Hz.
            
        Returns:
            Tuple of (List[TimebaseIssue], TimebaseStats).
        """
        stream_name = channel or "telemetry"
        if telemetry.empty or len(telemetry) < 2 or "timestamp" not in telemetry.columns:
            return (
                [],
                TimebaseStats(
                    mean_interval_s=0.0,
                    std_interval_s=0.0,
                    jitter_fraction=0.0,
                    sample_count=len(telemetry),
                ),
            )

        ts = pd.to_numeric(telemetry["timestamp"], errors="coerce").dropna().values
        if len(ts) < 2:
            return (
                [],
                TimebaseStats(
                    mean_interval_s=0.0,
                    std_interval_s=0.0,
                    jitter_fraction=0.0,
                    sample_count=len(ts),
                ),
            )

        issues: List[TimebaseIssue] = []
        deltas = np.diff(ts)

        # Monotonicity
        non_mono = np.where(deltas <= 0)[0]
        for idx in non_mono:
            issues.append(
                TimebaseIssue(
                    issue_type="non_monotonic",
                    severity="high",
                    time_range=(float(ts[idx]), float(ts[idx + 1])),
                    affected_stream=stream_name,
                    details=f"Telemetry non-monotonic timestamp at index {idx} ({ts[idx]:.4f}s -> {ts[idx + 1]:.4f}s)",
                )
            )

        # Expected delta
        if expected_rate_hz is not None and expected_rate_hz > 0:
            expected_interval = 1.0 / expected_rate_hz
        else:
            pos_deltas = deltas[deltas > 0]
            expected_interval = float(np.median(pos_deltas)) if len(pos_deltas) > 0 else 0.01

        if expected_interval > 0:
            jitter_mask = np.abs(deltas - expected_interval) / expected_interval >= self.config.jitter_threshold
            jitter_indices = np.where(jitter_mask)[0]
            jitter_fraction = float(np.mean(jitter_mask)) if len(deltas) > 0 else 0.0

            if jitter_fraction >= self.config.jitter_threshold and len(jitter_indices) > 0:
                issues.append(
                    TimebaseIssue(
                        issue_type="telemetry_jitter",
                        severity="medium",
                        time_range=(float(ts[jitter_indices[0]]), float(ts[jitter_indices[-1] + 1])),
                        affected_stream=stream_name,
                        details=(
                            f"Telemetry sampling jitter ({jitter_fraction:.2%}) exceeds threshold "
                            f"({self.config.jitter_threshold:.2%})"
                        ),
                    )
                )
        else:
            jitter_fraction = 0.0

        stats = TimebaseStats(
            mean_interval_s=float(np.mean(deltas)),
            std_interval_s=float(np.std(deltas)),
            jitter_fraction=jitter_fraction,
            sample_count=len(ts),
        )

        return issues, stats

    def check_cross_modal_sync(
        self,
        video_streams: Dict[str, List[Frame]],
        telemetry: pd.DataFrame,
    ) -> List[TimebaseIssue]:
        """Verify synchronization alignment between video frames and robot telemetry.
        
        Args:
            video_streams: Dictionary of camera streams with frames.
            telemetry: Telemetry DataFrame with 'timestamp' column in seconds.
            
        Returns:
            List of TimebaseIssue flagged when alignment exceeds tolerance (default 50ms).
        """
        issues: List[TimebaseIssue] = []
        if not video_streams or telemetry.empty or "timestamp" not in telemetry.columns:
            return issues

        telemetry_ts = pd.to_numeric(telemetry["timestamp"], errors="coerce").dropna().sort_values().values
        if len(telemetry_ts) == 0:
            return issues

        tolerance_s = self.config.sensor_desync_tolerance_ms / 1000.0

        for cam_id, frames in video_streams.items():
            if not frames:
                continue

            for frame in frames:
                video_t = frame.timestamp_us / 1_000_000.0

                # Binary search nearest telemetry timestamp
                idx = np.searchsorted(telemetry_ts, video_t)
                nearest_candidates = []
                if idx < len(telemetry_ts):
                    nearest_candidates.append(telemetry_ts[idx])
                if idx > 0:
                    nearest_candidates.append(telemetry_ts[idx - 1])

                if nearest_candidates:
                    min_delta = min(abs(video_t - c) for c in nearest_candidates)
                    if min_delta > tolerance_s:
                        issues.append(
                            TimebaseIssue(
                                issue_type="sensor_desync",
                                severity="medium",
                                time_range=(video_t, video_t + min_delta),
                                affected_stream=cam_id,
                                details=(
                                    f"Camera stream '{cam_id}' frame at {video_t:.3f}s lags nearest telemetry "
                                    f"by {min_delta * 1000.0:.1f}ms (threshold: {self.config.sensor_desync_tolerance_ms}ms)"
                                ),
                            )
                        )
                        # Report once per stream to avoid spamming
                        break

        return issues
