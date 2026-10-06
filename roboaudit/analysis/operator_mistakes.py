"""Operator Mistake Classifier for identifying teleoperation execution errors.

Detects:
- Object drops: Visual evidence of drop or missed drop anomaly (high severity) (Req 6.1)
- Alignment struggles: Approach alignment error > 15 degrees for > 2.0s (medium severity) (Req 6.2)
- Hesitations: Idle phase > 2.0s during active task (low severity) (Req 6.3)
- Fumbles: End-effector trajectory oscillation > 3 reversals/s (medium severity) (Req 6.4)

Annotates all mistakes with timestamp, severity, type, and camera evidence (Req 6.5)
and aggregates mistake summary statistics (Req 6.6).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from roboaudit.analysis.grasp import GraspAnomaly
from roboaudit.core.models import AuditConfig, Frame, OperatorMistake, Timeline


class OperatorMistakeClassifier:
    """Classifies human operator execution mistakes during robotic demonstrations."""

    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize classifier with audit configuration.
        
        Args:
            config: Optional audit configuration thresholds.
        """
        self.config = config or AuditConfig()
        self.hesitation_threshold_s = self.config.hesitation_threshold_s  # 2.0s default
        self.alignment_angle_threshold_deg = self.config.alignment_angle_threshold_deg  # 15.0 deg default
        self.reversals_threshold_per_s = 3.0  # Fumble oscillation threshold

    def classify_mistakes(
        self,
        telemetry: Optional[pd.DataFrame] = None,
        video_streams: Optional[Dict[str, List[Frame]]] = None,
        timeline: Optional[Timeline] = None,
        grasp_anomalies: Optional[List[GraspAnomaly]] = None,
        config: Optional[AuditConfig] = None,
    ) -> List[OperatorMistake]:
        """Classify operator mistakes from episode telemetry, timeline, and grasp anomalies.
        
        Args:
            telemetry: Telemetry DataFrame.
            video_streams: Optional dictionary of camera frames.
            timeline: Optional Timeline with classified TemporalWindows.
            grasp_anomalies: Optional pre-detected grasp anomalies.
            config: Optional override configuration.
            
        Returns:
            List of detected OperatorMistake instances.
        """
        cfg = config or self.config
        mistakes: List[OperatorMistake] = []

        # 1. Detect Drops (Requirement 6.1)
        # From grasp anomalies or telemetry flags
        if grasp_anomalies:
            for ga in grasp_anomalies:
                if ga.anomaly_type == "missed_drop":
                    mistakes.append(
                        OperatorMistake(
                            type="drop",
                            severity="high",
                            t_s=round(ga.timestamp_s, 2),
                            duration_s=None,
                            evidence=list(ga.visual_evidence) or ["frame_drop_detected"],
                        )
                    )

        if telemetry is not None and not telemetry.empty:
            df = telemetry.copy()
            if "timestamp" not in df.columns:
                if "timestamp_us" in df.columns:
                    df["timestamp"] = df["timestamp_us"] / 1_000_000.0
                elif "t" in df.columns:
                    df["timestamp"] = df["t"]
                else:
                    df["timestamp"] = np.arange(len(df)) * (1.0 / 30.0)

            # Check direct drop indicator columns
            for col in ["object_dropped", "drop_detected", "drop"]:
                if col in df.columns:
                    drop_rows = df[df[col].astype(bool)]
                    for _, row in drop_rows.iterrows():
                        t_s = float(row["timestamp"])
                        frame_num = int(round(t_s * 30.0))
                        mistakes.append(
                            OperatorMistake(
                                type="drop",
                                severity="high",
                                t_s=round(t_s, 2),
                                duration_s=None,
                                evidence=[f"camera_front:frame_{frame_num}"],
                            )
                        )

            # 2. Detect Alignment Struggles (Requirement 6.2)
            # Alignment error > 15 deg for > 2.0s during approach
            align_col = None
            for col in ["alignment_error_deg", "alignment_error", "angle_error"]:
                if col in df.columns:
                    align_col = col
                    break

            if align_col is not None:
                align_errs = df[align_col].to_numpy(dtype=float)
                ts = df["timestamp"].to_numpy(dtype=float)
                thresh_deg = self.alignment_angle_threshold_deg

                struggle_mask = align_errs > thresh_deg
                runs = self._find_contiguous_runs(struggle_mask)
                for start_idx, end_idx in runs:
                    dur = float(ts[end_idx] - ts[start_idx])
                    if dur > 2.0:
                        t_s = float(ts[start_idx])
                        frame_num = int(round(t_s * 30.0))
                        mean_err = float(np.mean(align_errs[start_idx : end_idx + 1]))
                        mistakes.append(
                            OperatorMistake(
                                type="alignment_struggle",
                                severity="medium",
                                t_s=round(t_s, 2),
                                duration_s=round(dur, 2),
                                evidence=[
                                    f"camera_front:frame_{frame_num}",
                                    f"alignment_error_{mean_err:.1f}deg_exceeded_{thresh_deg}deg_for_{dur:.1f}s",
                                ],
                            )
                        )

            # 3. Detect Fumbles (Requirement 6.4)
            # Trajectory velocity direction reversals > 3/s
            vel_col = None
            for col in ["velocity", "ee_velocity", "vel_x", "linear_velocity"]:
                if col in df.columns:
                    vel_col = col
                    break

            if vel_col is not None and len(df) >= 10:
                vels = df[vel_col].to_numpy(dtype=float)
                ts = df["timestamp"].to_numpy(dtype=float)
                signs = np.sign(np.diff(vels))
                sign_changes = np.diff(signs) != 0

                n_changes = len(sign_changes)
                window_size = min(30, n_changes)
                step = max(1, window_size // 2)

                for i in range(0, max(1, n_changes - window_size + 1), step):
                    sub_changes = sign_changes[i : i + window_size]
                    window_reversals = int(np.sum(sub_changes))
                    idx_end = min(len(ts) - 1, i + len(sub_changes))
                    window_dur = float(ts[idx_end] - ts[i])
                    if window_dur > 0 and (window_reversals / window_dur) > self.reversals_threshold_per_s:
                        t_s = float(ts[i])
                        frame_num = int(round(t_s * 30.0))
                        mistakes.append(
                            OperatorMistake(
                                type="fumble",
                                severity="medium",
                                t_s=round(t_s, 2),
                                duration_s=round(window_dur, 2),
                                evidence=[
                                    f"camera_front:frame_{frame_num}",
                                    f"oscillation_rate_{(window_reversals / window_dur):.1f}_reversals_per_s",
                                ],
                            )
                        )
                        break  # Report one per distinct fumbling segment

        # 4. Detect Hesitations (Requirement 6.3)
        # Idle phase > 2.0s during task execution
        if timeline is not None:
            for w in timeline.windows:
                dur = w.end_s - w.start_s
                if w.action_phase == "idle" and dur > self.hesitation_threshold_s:
                    frame_num = int(round(w.start_s * 30.0))
                    mistakes.append(
                        OperatorMistake(
                            type="hesitation",
                            severity="low",
                            t_s=round(w.start_s, 2),
                            duration_s=round(dur, 2),
                            evidence=[
                                f"camera_front:frame_{frame_num}",
                                f"idle_duration_{dur:.1f}s_exceeded_{self.hesitation_threshold_s}s",
                            ],
                        )
                    )

        # De-duplicate by (type, t_s rounded to 0.1s)
        deduped: List[OperatorMistake] = []
        seen = set()
        for m in mistakes:
            key = (m.type, round(m.t_s, 1))
            if key not in seen:
                seen.add(key)
                deduped.append(m)

        return deduped

    def aggregate_mistakes_summary(
        self, mistakes: List[OperatorMistake]
    ) -> Dict[str, Any]:
        """Aggregate operator mistakes into a summary breakdown.
        
        Requirement 6.6: Generate summary with counts by severity and type.
        
        Args:
            mistakes: List of OperatorMistake instances.
            
        Returns:
            Dictionary containing total count, counts by severity, and counts by type.
        """
        by_severity = {"low": 0, "medium": 0, "high": 0, "error": 0}
        by_type = {
            "drop": 0,
            "alignment_struggle": 0,
            "hesitation": 0,
            "fumble": 0,
            "collision": 0,
        }

        for m in mistakes:
            if m.severity in by_severity:
                by_severity[m.severity] += 1
            if m.type in by_type:
                by_type[m.type] += 1

        return {
            "total_mistakes": len(mistakes),
            "by_severity": by_severity,
            "by_type": by_type,
        }

    def _find_contiguous_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find start and end indices of contiguous True values in a boolean array."""
        runs: List[Tuple[int, int]] = []
        in_run = False
        start_idx = 0

        for i, val in enumerate(mask):
            if val and not in_run:
                in_run = True
                start_idx = i
            elif not val and in_run:
                in_run = False
                runs.append((start_idx, i - 1))

        if in_run:
            runs.append((start_idx, len(mask) - 1))

        return runs
