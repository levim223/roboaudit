"""Action Phase Segmenter for temporal decomposition and timeline reconstruction.

Decomposes robotics demonstrations into semantic action phases:
- approach: Gripper open + end-effector motion toward target
- grasp: State transition open -> closed
- manipulate: Gripper closed + end-effector motion
- release: State transition closed -> open
- idle: End-effector velocity below threshold for extended duration

Also classifies contribution (advancing, wasteful, idle), ensures progress monotonicity,
and detects operator hesitations (> 5.0s without progress).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from roboaudit.core.models import (
    AuditConfig,
    Frame,
    OperatorMistake,
    TemporalWindow,
    Timeline,
)


class ActionPhaseSegmenter:
    """Segments robot telemetry and videos into semantic temporal action windows."""

    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize segmenter with audit configuration.
        
        Args:
            config: Optional audit configuration thresholds.
        """
        self.config = config or AuditConfig()
        self.velocity_threshold = 0.05
        self.gripper_closed_thresh = 0.70
        self.gripper_open_thresh = 0.30
        self.idle_duration_thresh_s = self.config.hesitation_threshold_s  # 2.0s
        self.hesitation_thresh_s = 5.0  # Requirement 4.7

    def segment_episode(
        self,
        telemetry: pd.DataFrame,
        video_streams: Optional[Dict[str, List[Frame]]] = None,
        config: Optional[AuditConfig] = None,
    ) -> Timeline:
        """Segment an episode's telemetry into a Timeline of TemporalWindows.
        
        Args:
            telemetry: Telemetry DataFrame containing timestamps and sensor signals.
            video_streams: Optional dictionary of extracted video frames per camera.
            config: Optional override configuration.
            
        Returns:
            Timeline containing ordered list of TemporalWindows.
        """
        cfg = config or self.config

        if telemetry.empty:
            return Timeline(windows=[])

        df = telemetry.copy()
        if "timestamp" not in df.columns:
            if "timestamp_us" in df.columns:
                df["timestamp"] = df["timestamp_us"] / 1_000_000.0
            elif "t" in df.columns:
                df["timestamp"] = df["t"]
            else:
                df["timestamp"] = np.arange(len(df)) * (1.0 / 30.0)

        df = df.sort_values("timestamp").reset_index(drop=True)

        if len(df) < 2:
            # Single sample: minimal single window
            t0 = float(df["timestamp"].iloc[0])
            t1 = t0 + 1.0
            window = TemporalWindow(
                start_s=t0,
                end_s=t1,
                action_phase="idle",
                arm_attribution="none",
                contribution_type="idle",
                completion_percentage=0.0,
            )
            return Timeline(windows=[window])

        # Extract velocities and gripper states
        velocities = self._extract_velocities(df)
        gripper_states = self._extract_gripper_states(df)
        arms = self._extract_arm_attributions(df)

        timestamps = df["timestamp"].to_numpy(dtype=float)
        duration = float(timestamps[-1] - timestamps[0])

        if duration <= 0:
            duration = 1.0
            timestamps = np.linspace(0.0, 1.0, len(df))

        # Sample or segment into intervals (target window length ~1.0s to 3.0s or state changes)
        change_points = self._find_phase_change_points(
            timestamps, velocities, gripper_states
        )

        windows: List[TemporalWindow] = []
        cum_progress = 0.0
        n_segments = len(change_points) - 1

        # Count how many segments will be advancing to apportion progress monotonically
        advancing_count = 0
        tentative_phases = []
        for i in range(n_segments):
            idx_start = change_points[i]
            idx_end = change_points[i + 1]
            sub_vel = velocities[idx_start:idx_end]
            sub_grip = gripper_states[idx_start:idx_end]
            dt = timestamps[idx_end - 1] - timestamps[idx_start]

            phase = self._classify_phase_window(sub_vel, sub_grip, dt)
            contrib = "idle" if phase == "idle" else "advancing"
            tentative_phases.append((phase, contrib))
            if contrib == "advancing":
                advancing_count += 1

        progress_step = (1.0 / max(1, advancing_count)) if advancing_count > 0 else 0.0

        for i in range(n_segments):
            idx_start = change_points[i]
            idx_end = change_points[i + 1]

            t_start = round(float(timestamps[idx_start]), 3)
            t_end = round(float(timestamps[idx_end if idx_end < len(timestamps) else -1]), 3)

            # Ensure strict end > start
            if t_end <= t_start:
                t_end = round(t_start + 0.1, 3)

            phase, contrib = tentative_phases[i]
            arm = arms[idx_start] if phase != "idle" else "none"

            if contrib == "advancing":
                cum_progress = min(1.0, round(cum_progress + progress_step, 3))
            
            # Create window
            w = TemporalWindow(
                start_s=t_start,
                end_s=t_end,
                action_phase=phase,
                arm_attribution=arm,
                contribution_type=contrib,
                completion_percentage=min(1.0, max(0.0, cum_progress)),
            )
            windows.append(w)

        # Enforce last window reaches 1.0 if any advancing occurred
        if advancing_count > 0 and windows:
            # Set final advancing window to 1.0
            for w in reversed(windows):
                if w.contribution_type == "advancing":
                    w.completion_percentage = 1.0
                    break

        timeline = Timeline(windows=windows)
        return timeline

    def classify_contribution(
        self,
        window: TemporalWindow,
        telemetry_slice: Optional[pd.DataFrame] = None,
    ) -> Literal["advancing", "wasteful", "idle"]:
        """Classify contribution type for a temporal window.
        
        Args:
            window: TemporalWindow to evaluate.
            telemetry_slice: Optional telemetry slice during the window.
            
        Returns:
            One of 'advancing', 'wasteful', 'idle'.
        """
        if window.action_phase == "idle":
            return "idle"
        
        if telemetry_slice is not None and not telemetry_slice.empty:
            # Check for oscillatory / wasteful motion
            if "velocity" in telemetry_slice.columns:
                vel = telemetry_slice["velocity"].to_numpy()
                if np.std(vel) > 1.5 * np.mean(vel) and np.mean(vel) < self.velocity_threshold:
                    return "wasteful"

        return "advancing"

    def calculate_completion(
        self,
        window: TemporalWindow,
        timeline_context: List[TemporalWindow],
    ) -> float:
        """Estimate completion percentage for a window given preceding context.
        
        Args:
            window: Current TemporalWindow.
            timeline_context: List of prior TemporalWindows.
            
        Returns:
            Completion percentage in [0.0, 1.0].
        """
        if not timeline_context:
            return 0.1 if window.contribution_type == "advancing" else 0.0

        prev_comp = timeline_context[-1].completion_percentage
        if window.contribution_type == "advancing":
            return min(1.0, round(prev_comp + 0.1, 3))
        return prev_comp

    def detect_hesitations(
        self,
        timeline: Timeline,
        threshold_s: Optional[float] = None,
    ) -> List[OperatorMistake]:
        """Detect operator hesitations persisting longer than threshold without progress.
        
        Requirement 4.7: Flag hesitation warning if an Action_Phase persists > 5.0s
        without completion progress.
        
        Args:
            timeline: Audited timeline.
            threshold_s: Duration threshold in seconds (default: 5.0s).
            
        Returns:
            List of OperatorMistake entries of type 'hesitation'.
        """
        thresh = threshold_s if threshold_s is not None else self.hesitation_thresh_s
        hesitations: List[OperatorMistake] = []

        if not timeline.windows:
            return hesitations

        # 1. Check individual window durations
        prev_progress = 0.0
        for w in timeline.windows:
            dur = w.end_s - w.start_s
            progress_delta = w.completion_percentage - prev_progress
            
            if dur > thresh and (progress_delta <= 0.001 or w.action_phase == "idle"):
                hesitations.append(
                    OperatorMistake(
                        type="hesitation",
                        severity="low",
                        t_s=round(w.start_s, 2),
                        duration_s=round(dur, 2),
                        evidence=[
                            f"Phase '{w.action_phase}' persisted {dur:.1f}s without progress (threshold: {thresh}s)"
                        ],
                    )
                )
            prev_progress = w.completion_percentage

        # 2. Check contiguous spans of the same phase with no progress
        span_start = timeline.windows[0].start_s
        span_phase = timeline.windows[0].action_phase
        span_initial_comp = timeline.windows[0].completion_percentage
        last_end = timeline.windows[0].end_s

        for w in timeline.windows[1:]:
            if w.action_phase == span_phase:
                last_end = w.end_s
            else:
                span_dur = last_end - span_start
                comp_delta = w.completion_percentage - span_initial_comp
                if span_dur > thresh and comp_delta <= 0.001:
                    # Avoid duplicate if already caught by individual window
                    if not any(h.t_s == round(span_start, 2) for h in hesitations):
                        hesitations.append(
                            OperatorMistake(
                                type="hesitation",
                                severity="low",
                                t_s=round(span_start, 2),
                                duration_s=round(span_dur, 2),
                                evidence=[
                                    f"Contiguous phase '{span_phase}' persisted {span_dur:.1f}s without progress"
                                ],
                            )
                        )
                span_start = w.start_s
                span_phase = w.action_phase
                span_initial_comp = w.completion_percentage
                last_end = w.end_s

        span_dur = last_end - span_start
        if span_dur > thresh:
            if not any(h.t_s == round(span_start, 2) for h in hesitations):
                hesitations.append(
                    OperatorMistake(
                        type="hesitation",
                        severity="low",
                        t_s=round(span_start, 2),
                        duration_s=round(span_dur, 2),
                        evidence=[
                            f"Contiguous phase '{span_phase}' persisted {span_dur:.1f}s without progress"
                        ],
                    )
                )

        return hesitations

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _extract_velocities(self, df: pd.DataFrame) -> np.ndarray:
        """Extract or estimate end-effector / joint velocities."""
        if "velocity" in df.columns:
            return np.abs(df["velocity"].to_numpy(dtype=float))
        if "ee_velocity" in df.columns:
            return np.abs(df["ee_velocity"].to_numpy(dtype=float))

        # Check for position columns
        pos_cols = [c for c in ["ee_pos_x", "ee_pos_y", "ee_pos_z"] if c in df.columns]
        if pos_cols:
            pos = df[pos_cols].to_numpy(dtype=float)
            dt = np.diff(df["timestamp"].to_numpy(dtype=float), prepend=df["timestamp"].iloc[0] - 0.033)
            dt[dt <= 0] = 0.033
            vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[[0]]), axis=1) / dt
            return vel

        # Check for joint positions
        joint_cols = [c for c in df.columns if c.startswith("joint_") or c.startswith("q_")]
        if joint_cols:
            q = df[joint_cols].to_numpy(dtype=float)
            dt = np.diff(df["timestamp"].to_numpy(dtype=float), prepend=df["timestamp"].iloc[0] - 0.033)
            dt[dt <= 0] = 0.033
            diff = np.diff(q, axis=0, prepend=q[[0]])
            vel = np.sum(np.abs(diff), axis=1) / dt
            return vel

        # Default fallback: small nominal motion
        return np.full(len(df), 0.1, dtype=float)

    def _extract_gripper_states(self, df: pd.DataFrame) -> np.ndarray:
        """Extract gripper state in [0.0 (open), 1.0 (closed)]."""
        for col in ["gripper_state", "gripper_position", "gripper_width", "gripper"]:
            if col in df.columns:
                g = df[col].to_numpy(dtype=float)
                # Normalize if outside [0.0, 1.0]
                g_min, g_max = np.min(g), np.max(g)
                if g_max > 1.0 or g_min < 0.0:
                    g = (g - g_min) / max(1e-6, g_max - g_min)
                return g
        return np.zeros(len(df), dtype=float)

    def _extract_arm_attributions(
        self, df: pd.DataFrame
    ) -> List[Literal["left", "right", "both", "none"]]:
        """Extract arm attribution per row."""
        if "arm" in df.columns:
            valid = {"left", "right", "both", "none"}
            return [
                val if val in valid else "right"
                for val in df["arm"].astype(str)
            ]
        if "arm_attribution" in df.columns:
            valid = {"left", "right", "both", "none"}
            return [
                val if val in valid else "right"
                for val in df["arm_attribution"].astype(str)
            ]
        # Single-arm default
        return ["right"] * len(df)

    def _find_phase_change_points(
        self,
        timestamps: np.ndarray,
        velocities: np.ndarray,
        gripper_states: np.ndarray,
        target_window_duration_s: float = 2.0,
    ) -> List[int]:
        """Identify indices where action phases change or uniform boundaries occur."""
        n = len(timestamps)
        if n <= 2:
            return [0, n]

        change_points = {0, n}

        # 1. State changes: gripper opening/closing
        grip_diff = np.diff(gripper_states)
        for i, delta in enumerate(grip_diff):
            if abs(delta) > 0.3:  # Significant gripper movement
                change_points.add(i)
                if i + 1 < n:
                    change_points.add(i + 1)

        # 2. Motion changes: transition between idle and active
        is_moving = velocities >= self.velocity_threshold
        motion_diff = np.diff(is_moving.astype(int))
        for i, delta in enumerate(motion_diff):
            if delta != 0:
                change_points.add(i)
                if i + 1 < n:
                    change_points.add(i + 1)

        # 3. Time-based boundaries every ~2.0 seconds for moving phases
        t_start = timestamps[0]
        cur_target = t_start + target_window_duration_s
        for i, t in enumerate(timestamps):
            if t >= cur_target:
                if is_moving[i]:
                    change_points.add(i)
                cur_target = t + target_window_duration_s

        sorted_pts = sorted(list(change_points))
        # Filter points that are too close together (< 2 samples)
        filtered = [sorted_pts[0]]
        for pt in sorted_pts[1:]:
            if pt - filtered[-1] >= 2 or pt == n:
                filtered.append(pt)

        if filtered[-1] != n:
            filtered.append(n)

        return filtered

    def _classify_phase_window(
        self,
        velocities: np.ndarray,
        gripper_states: np.ndarray,
        duration_s: float,
    ) -> Literal["approach", "grasp", "manipulate", "release", "idle"]:
        """Classify a slice of sensor readings into an ActionPhase."""
        mean_vel = float(np.mean(velocities)) if len(velocities) > 0 else 0.0
        max_vel = float(np.max(velocities)) if len(velocities) > 0 else 0.0
        start_grip = float(gripper_states[0]) if len(gripper_states) > 0 else 0.0
        end_grip = float(gripper_states[-1]) if len(gripper_states) > 0 else 0.0
        mean_grip = float(np.mean(gripper_states)) if len(gripper_states) > 0 else 0.0

        # Idle heuristic: low velocity for >= 2.0s, or entire segment is below threshold
        if mean_vel < self.velocity_threshold:
            if duration_s >= 1.8 or max_vel < self.velocity_threshold:
                return "idle"

        # Grasp heuristic: open -> closed transition
        if start_grip < self.gripper_open_thresh and end_grip >= self.gripper_closed_thresh:
            return "grasp"
        if end_grip - start_grip > 0.4:
            return "grasp"

        # Release heuristic: closed -> open transition
        if start_grip >= self.gripper_closed_thresh and end_grip < self.gripper_open_thresh:
            return "release"
        if start_grip - end_grip > 0.4:
            return "release"

        # Manipulate: gripper closed + moving
        if mean_grip >= self.gripper_closed_thresh:
            return "manipulate"

        # Approach: gripper open + moving
        return "approach"
