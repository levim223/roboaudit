"""Grasp anomaly detection and visual-sensor cross-referencing.

Detects contradictions between gripper sensor states and visual evidence:
- Phantom grasp: gripper motor closed, but object elevation delta is zero (high severity)
- Missed drop: gripper registered closed, but visual evidence indicates object falling (high severity)
- Sensor mismatch: gripper force > 0, but no object visible in gripper bounds (medium severity)

Annotates all anomalies with camera evidence references and calculates object elevation
via optical flow across frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from roboaudit.core.models import AuditConfig, DataIssue, Frame


@dataclass
class GraspAnomaly:
    """A detected contradiction between gripper state and visual observation.
    
    Attributes:
        anomaly_type: Type of anomaly ('phantom_grasp', 'missed_drop', 'sensor_mismatch')
        severity: Anomaly severity ('low', 'medium', 'high', 'error')
        timestamp_s: Timestamp in seconds
        gripper_state: Description or value of gripper state
        visual_evidence: List of camera frame identifiers and bounding details
        details: Human-readable diagnostic description
    """
    anomaly_type: Literal["phantom_grasp", "missed_drop", "sensor_mismatch"]
    severity: Literal["low", "medium", "high", "error"]
    timestamp_s: float
    gripper_state: str
    visual_evidence: List[str] = field(default_factory=list)
    details: str = ""

    def to_data_issue(self) -> DataIssue:
        """Convert GraspAnomaly to standard DataIssue for reporting."""
        return DataIssue(
            issue=f"{self.anomaly_type}: {self.details}",
            category="grasp",
            severity=self.severity,
            t_s=round(self.timestamp_s, 2),
            evidence=list(self.visual_evidence),
        )


class GraspAnomalyDetector:
    """Detects contradictions between gripper telemetry and camera evidence."""

    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize detector with audit configuration.
        
        Args:
            config: Optional audit configuration thresholds.
        """
        self.config = config or AuditConfig()
        self.gripper_closed_thresh = 0.70
        self.force_threshold = 0.5  # Newtons / threshold

    def detect_anomalies(
        self,
        telemetry: pd.DataFrame,
        video_streams: Optional[Dict[str, List[Frame]]] = None,
        config: Optional[AuditConfig] = None,
        elevation_deltas: Optional[Dict[float, float]] = None,
        falling_detections: Optional[Dict[float, bool]] = None,
        bbox_occupancies: Optional[Dict[float, bool]] = None,
    ) -> List[GraspAnomaly]:
        """Detect grasp contradictions cross-referencing sensors and vision.
        
        Args:
            telemetry: Telemetry DataFrame with timestamp, gripper_state, and optional force.
            video_streams: Optional dictionary of camera streams with Frame objects.
            config: Optional override configuration.
            elevation_deltas: Optional mapping of timestamp to measured elevation delta.
            falling_detections: Optional mapping of timestamp to object falling boolean.
            bbox_occupancies: Optional mapping of timestamp to whether object is in gripper bbox.
            
        Returns:
            List of detected GraspAnomaly instances.
        """
        anomalies: List[GraspAnomaly] = []

        if telemetry.empty:
            return anomalies

        df = telemetry.copy()
        if "timestamp" not in df.columns:
            if "timestamp_us" in df.columns:
                df["timestamp"] = df["timestamp_us"] / 1_000_000.0
            elif "t" in df.columns:
                df["timestamp"] = df["t"]
            else:
                df["timestamp"] = np.arange(len(df)) * (1.0 / 30.0)

        gripper_col = None
        for col in ["gripper_state", "gripper_position", "gripper_width", "gripper"]:
            if col in df.columns:
                gripper_col = col
                break

        if gripper_col is None:
            return anomalies

        force_col = None
        for col in ["gripper_force", "contact_force", "force"]:
            if col in df.columns:
                force_col = col
                break

        timestamps = df["timestamp"].to_numpy(dtype=float)
        gripper_vals = df[gripper_col].to_numpy(dtype=float)
        forces = df[force_col].to_numpy(dtype=float) if force_col else np.zeros(len(df))

        # Check each record for anomalies
        for i, (t, grip, f_val) in enumerate(zip(timestamps, gripper_vals, forces)):
            t_s = float(t)
            is_closed = grip >= self.gripper_closed_thresh

            # Frame label reference
            frame_ref = self._find_nearest_frame_ref(t_s, video_streams)

            # 1. Phantom Grasp Detection (Requirement 3.1)
            # Motor closed, but visual elevation delta is zero
            elev_delta = None
            if elevation_deltas is not None and t_s in elevation_deltas:
                elev_delta = elevation_deltas[t_s]
            elif "elevation_delta" in df.columns:
                elev_delta = float(df["elevation_delta"].iloc[i])

            if is_closed and elev_delta is not None and abs(elev_delta) < 1e-4:
                anomalies.append(
                    GraspAnomaly(
                        anomaly_type="phantom_grasp",
                        severity="high",
                        timestamp_s=round(t_s, 3),
                        gripper_state=f"closed (value={grip:.2f})",
                        visual_evidence=[frame_ref],
                        details=(
                            f"Gripper closed ({grip:.2f}) at t={t_s:.2f}s but object elevation "
                            f"delta is zero ({elev_delta:.4f}m)"
                        ),
                    )
                )

            # 2. Missed Drop Detection (Requirement 3.3)
            # Motor closed, but visual evidence shows object falling
            is_falling = False
            if falling_detections is not None and t_s in falling_detections:
                is_falling = falling_detections[t_s]
            elif "object_falling" in df.columns:
                is_falling = bool(df["object_falling"].iloc[i])

            if is_closed and is_falling:
                anomalies.append(
                    GraspAnomaly(
                        anomaly_type="missed_drop",
                        severity="high",
                        timestamp_s=round(t_s, 3),
                        gripper_state=f"closed (value={grip:.2f})",
                        visual_evidence=[frame_ref],
                        details=(
                            f"Gripper closed ({grip:.2f}) at t={t_s:.2f}s while visual tracking "
                            f"detected object dropping"
                        ),
                    )
                )

            # 3. Sensor Mismatch Detection (Requirement 3.2)
            # Force sensor > 0, but no object within gripper bounding box
            has_force = f_val > self.force_threshold
            has_object_in_bbox = True
            if bbox_occupancies is not None and t_s in bbox_occupancies:
                has_object_in_bbox = bbox_occupancies[t_s]
            elif "object_in_gripper" in df.columns:
                has_object_in_bbox = bool(df["object_in_gripper"].iloc[i])

            if has_force and not has_object_in_bbox:
                anomalies.append(
                    GraspAnomaly(
                        anomaly_type="sensor_mismatch",
                        severity="medium",
                        timestamp_s=round(t_s, 3),
                        gripper_state=f"contact_force={f_val:.2f}N",
                        visual_evidence=[frame_ref],
                        details=(
                            f"Gripper force sensor measured contact ({f_val:.2f}N) but visual inspection "
                            f"confirmed no object within gripper bounds"
                        ),
                    )
                )

        return anomalies

    def measure_object_elevation(
        self,
        frames: List[Frame],
        gripper_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """Calculate object elevation delta across frames using optical flow.
        
        Args:
            frames: Chronological sequence of Frame objects.
            gripper_bbox: Optional bounding box (x, y, w, h) around gripper.
            
        Returns:
            Measured elevation delta in pixels or normalized units (positive = upward).
        """
        if len(frames) < 2:
            return 0.0

        try:
            import cv2
        except ImportError:
            # Fallback if cv2 is not available
            return 0.0

        prev_frame = frames[0].image
        curr_frame = frames[-1].image

        if prev_frame is None or curr_frame is None:
            return 0.0

        # Convert to grayscale
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            curr_gray = curr_frame

        # Crop to gripper bbox if provided
        if gripper_bbox is not None:
            x, y, w, h = gripper_bbox
            h_img, w_img = prev_gray.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)
            if x2 > x1 and y2 > y1:
                prev_gray = prev_gray[y1:y2, x1:x2]
                curr_gray = curr_gray[y1:y2, x1:x2]

        if prev_gray.size == 0 or curr_gray.size == 0:
            return 0.0

        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # In image coordinates, y increases downward.
        # Elevation delta: negative y flow corresponds to upward movement
        vy = flow[..., 1]
        elevation_delta = float(-np.median(vy))
        return elevation_delta

    def _find_nearest_frame_ref(
        self,
        timestamp_s: float,
        video_streams: Optional[Dict[str, List[Frame]]],
    ) -> str:
        """Find the nearest camera frame reference for a given timestamp."""
        if not video_streams:
            frame_num = int(round(timestamp_s * 30.0))
            return f"frame_{frame_num}"

        target_us = int(timestamp_s * 1_000_000)
        best_cam = next(iter(video_streams))
        best_diff = float("inf")
        best_pts = 0

        for cam_id, frames in video_streams.items():
            for frame in frames:
                diff = abs(frame.timestamp_us - target_us)
                if diff < best_diff:
                    best_diff = diff
                    best_cam = cam_id
                    best_pts = frame.pts

        return f"{best_cam}_pts_{best_pts}"
