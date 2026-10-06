"""Sensor Dropout Monitor for detecting communication gaps, frozen channels, and outliers.

Detects:
- Telemetry dropouts exceeding 100ms gap threshold (Req 11.1, 11.5)
- Frozen sensors with constant values > 1.0s during motion phases (Req 11.2)
- Sensor outliers violating configured physical/joint limits (Req 11.3, 11.4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from roboaudit.core.models import AuditConfig, DataIssue


@dataclass
class DropoutIssue:
    """A detected sensor dropout, freeze, or range outlier issue.
    
    Attributes:
        issue_type: Type of issue ('sensor_dropout', 'sensor_freeze', 'sensor_outlier')
        severity: Severity level ('warning', 'error', 'low', 'medium', 'high')
        channel: Affected telemetry sensor channel
        time_range: Start and end timestamp of occurrence (start_s, end_s)
        duration_s: Total duration in seconds
        percentage_of_episode: Percentage of episode duration affected
        details: Human-readable diagnostic description
    """
    issue_type: Literal["sensor_dropout", "sensor_freeze", "sensor_outlier"]
    severity: Literal["warning", "error", "low", "medium", "high"]
    channel: str
    time_range: Tuple[float, float]
    duration_s: float
    percentage_of_episode: float
    details: str

    def to_data_issue(self) -> DataIssue:
        """Convert DropoutIssue to standard DataIssue."""
        sev = "high" if self.severity in {"error", "high"} else "medium"
        return DataIssue(
            issue=f"{self.issue_type}: {self.details}",
            category="sensor",
            severity=sev,
            t_s=round(self.time_range[0], 2),
            evidence=[f"{self.channel}_duration_{self.duration_s:.2f}s"],
        )


class SensorDropoutMonitor:
    """Monitors telemetry streams for dropouts, frozen signals, and physical outliers."""

    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize monitor with audit configuration thresholds.
        
        Args:
            config: Optional audit configuration thresholds.
        """
        self.config = config or AuditConfig()
        self.gap_threshold_ms = self.config.sensor_gap_threshold_ms  # 100.0ms default
        self.freeze_threshold_s = 1.0  # 1.0s default
        self.default_joint_limits = (-2.0 * np.pi, 2.0 * np.pi)
        self.default_force_limits = (-100.0, 100.0)

    def detect_dropouts(
        self,
        telemetry: pd.DataFrame,
        gap_threshold_ms: Optional[float] = None,
    ) -> List[DropoutIssue]:
        """Detect timestamp communication dropouts exceeding the gap threshold.
        
        Requirements 11.1, 11.5: Flag sensor_dropout warning with affected channel,
        time range, duration, and percentage of episode affected.
        
        Args:
            telemetry: Telemetry DataFrame with timestamp column.
            gap_threshold_ms: Threshold in ms (defaults to 100.0ms).
            
        Returns:
            List of detected DropoutIssue instances of type 'sensor_dropout'.
        """
        thresh_ms = gap_threshold_ms if gap_threshold_ms is not None else self.gap_threshold_ms
        thresh_s = thresh_ms / 1000.0
        issues: List[DropoutIssue] = []

        if telemetry.empty:
            return issues

        df = telemetry.copy()
        if "timestamp" not in df.columns:
            if "timestamp_us" in df.columns:
                df["timestamp"] = df["timestamp_us"] / 1_000_000.0
            elif "t" in df.columns:
                df["timestamp"] = df["t"]
            else:
                return issues

        df = df.sort_values("timestamp").reset_index(drop=True)
        ts = df["timestamp"].to_numpy(dtype=float)

        if len(ts) < 2:
            return issues

        ep_duration = max(1e-6, float(ts[-1] - ts[0]))
        deltas = np.diff(ts)

        for i, dt in enumerate(deltas):
            if dt > thresh_s:
                t_start = float(ts[i])
                t_end = float(ts[i + 1])
                gap_dur = float(dt)
                pct = round((gap_dur / ep_duration) * 100.0, 2)

                issues.append(
                    DropoutIssue(
                        issue_type="sensor_dropout",
                        severity="warning",
                        channel="telemetry_bus",
                        time_range=(round(t_start, 3), round(t_end, 3)),
                        duration_s=round(gap_dur, 3),
                        percentage_of_episode=pct,
                        details=(
                            f"Telemetry gap of {gap_dur * 1000.0:.1f}ms detected between "
                            f"t={t_start:.3f}s and t={t_end:.3f}s (threshold: {thresh_ms:.1f}ms, "
                            f"{pct:.1f}% of episode)"
                        ),
                    )
                )

        return issues

    def detect_frozen_sensors(
        self,
        telemetry: pd.DataFrame,
        freeze_duration_s: Optional[float] = None,
    ) -> List[DropoutIssue]:
        """Detect sensors whose readings remain completely static while motion occurs.
        
        Requirement 11.2: Flag sensor_freeze warning when values remain constant
        for more than 1.0 second during motion phases.
        
        Args:
            telemetry: Telemetry DataFrame with timestamp and sensor channels.
            freeze_duration_s: Minimum frozen duration in seconds (defaults to 1.0s).
            
        Returns:
            List of detected DropoutIssue instances of type 'sensor_freeze'.
        """
        dur_thresh = freeze_duration_s if freeze_duration_s is not None else self.freeze_threshold_s
        issues: List[DropoutIssue] = []

        if telemetry.empty:
            return issues

        df = telemetry.copy()
        if "timestamp" not in df.columns:
            if "timestamp_us" in df.columns:
                df["timestamp"] = df["timestamp_us"] / 1_000_000.0
            elif "t" in df.columns:
                df["timestamp"] = df["t"]
            else:
                return issues

        df = df.sort_values("timestamp").reset_index(drop=True)
        ts = df["timestamp"].to_numpy(dtype=float)
        if len(ts) < 5:
            return issues

        ep_duration = max(1e-6, float(ts[-1] - ts[0]))

        # Identify channels to check (numeric columns excluding timestamp)
        skip_cols = {"timestamp", "timestamp_us", "timestamp_ms", "t", "time_s", "frame_idx", "step"}
        candidate_cols = [
            c for c in df.columns
            if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Check if the episode as a whole is in motion (some channel is changing)
        for col in candidate_cols:
            vals = df[col].to_numpy(dtype=float)
            # Find contiguous runs of constant values
            run_start_idx = 0
            for i in range(1, len(vals)):
                val_diff = abs(vals[i] - vals[run_start_idx])
                if val_diff > 1e-6 or i == len(vals) - 1:
                    end_idx = i if val_diff > 1e-6 else len(vals) - 1
                    run_dur = float(ts[end_idx] - ts[run_start_idx])

                    if run_dur > dur_thresh:
                        # Verify that other channels were moving during this run
                        sub_df = df.iloc[run_start_idx:end_idx]
                        other_cols = [c for c in candidate_cols if c != col]
                        is_motion = False
                        for oc in other_cols:
                            if np.ptp(sub_df[oc].to_numpy(dtype=float)) > 0.05:
                                is_motion = True
                                break

                        if is_motion:
                            pct = round((run_dur / ep_duration) * 100.0, 2)
                            t_start = float(ts[run_start_idx])
                            t_end = float(ts[end_idx])
                            issues.append(
                                DropoutIssue(
                                    issue_type="sensor_freeze",
                                    severity="warning",
                                    channel=col,
                                    time_range=(round(t_start, 3), round(t_end, 3)),
                                    duration_s=round(run_dur, 3),
                                    percentage_of_episode=pct,
                                    details=(
                                        f"Channel '{col}' frozen at constant value ({vals[run_start_idx]:.4f}) "
                                        f"for {run_dur:.2f}s during active motion (threshold: {dur_thresh:.1f}s)"
                                    ),
                                )
                            )
                    run_start_idx = i

        return issues

    def detect_outliers(
        self,
        telemetry: pd.DataFrame,
        joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        force_limits: Optional[Tuple[float, float]] = None,
    ) -> List[DropoutIssue]:
        """Validate telemetry values against physically plausible ranges.
        
        Requirements 11.3, 11.4: Flag sensor_outlier warning for joint or force
        readings exceeding configured bounds.
        
        Args:
            telemetry: Telemetry DataFrame.
            joint_limits: Optional dictionary of joint name -> (min_val, max_val).
            force_limits: Optional tuple (min_force, max_force).
            
        Returns:
            List of detected DropoutIssue instances of type 'sensor_outlier'.
        """
        limits = joint_limits or (self.config.joint_limits or {})
        f_lims = force_limits or self.default_force_limits
        issues: List[DropoutIssue] = []

        if telemetry.empty:
            return issues

        df = telemetry.copy()
        if "timestamp" not in df.columns:
            ts = np.arange(len(df)) * (1.0 / 30.0)
        else:
            ts = df["timestamp"].to_numpy(dtype=float)

        ep_duration = max(1e-6, float(ts[-1] - ts[0])) if len(ts) > 1 else 1.0

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            vals = df[col].to_numpy(dtype=float)

            # Determine limit range for this column
            if col in limits:
                c_min, c_max = limits[col]
            elif "force" in col:
                c_min, c_max = f_lims
            elif col.startswith("joint_") or col.startswith("q_"):
                c_min, c_max = self.default_joint_limits
            else:
                continue

            outlier_mask = (vals < c_min) | (vals > c_max)
            if np.any(outlier_mask):
                outlier_indices = np.where(outlier_mask)[0]
                t_first = float(ts[outlier_indices[0]])
                t_last = float(ts[outlier_indices[-1]])
                dur = max(0.01, t_last - t_first)
                pct = round((dur / ep_duration) * 100.0, 2)
                min_seen, max_seen = float(np.min(vals[outlier_mask])), float(np.max(vals[outlier_mask]))

                issues.append(
                    DropoutIssue(
                        issue_type="sensor_outlier",
                        severity="warning",
                        channel=col,
                        time_range=(round(t_first, 3), round(t_last, 3)),
                        duration_s=round(dur, 3),
                        percentage_of_episode=pct,
                        details=(
                            f"Channel '{col}' values outside valid range [{c_min:.2f}, {c_max:.2f}]: "
                            f"observed [{min_seen:.2f}, {max_seen:.2f}] across {len(outlier_indices)} samples"
                        ),
                    )
                )

        return issues

    def monitor_all(
        self,
        telemetry: pd.DataFrame,
    ) -> List[DropoutIssue]:
        """Run all sensor health checks and return aggregated issues."""
        issues: List[DropoutIssue] = []
        issues.extend(self.detect_dropouts(telemetry))
        issues.extend(self.detect_frozen_sensors(telemetry))
        issues.extend(self.detect_outliers(telemetry))
        return issues
