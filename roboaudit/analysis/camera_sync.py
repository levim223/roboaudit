"""Multi-camera synchronization verification and stream swap detection.

Verifies:
- Frame timestamp alignment across cameras within 16ms tolerance (Req 10.1, 10.2)
- Frame count consistency across camera streams within 2-frame tolerance (Req 10.3)
- Camera stream identifier swap detection via feature consistency (Req 10.4, 10.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np

from roboaudit.core.models import AuditConfig, DataIssue, Frame


@dataclass
class CameraSyncIssue:
    """A detected multi-camera synchronization issue or stream swap.
    
    Attributes:
        issue_type: Type of issue ('desync', 'frame_count_mismatch', 'camera_swap')
        severity: Severity level ('warning', 'error')
        camera_pair: Tuple of affected camera identifiers (cam_a, cam_b)
        time_range: Optional temporal bounds (start_s, end_s)
        details: Human-readable diagnostic description
    """
    issue_type: Literal["desync", "frame_count_mismatch", "camera_swap"]
    severity: Literal["warning", "error"]
    camera_pair: Tuple[str, str]
    time_range: Optional[Tuple[float, float]] = None
    details: str = ""

    def to_data_issue(self) -> DataIssue:
        """Convert CameraSyncIssue to standard DataIssue."""
        t_s = self.time_range[0] if self.time_range else 0.0
        sev = "high" if self.severity == "error" else "medium"
        return DataIssue(
            issue=f"camera_{self.issue_type}: {self.details}",
            category="camera",
            severity=sev,
            t_s=round(t_s, 2),
            evidence=[f"{self.camera_pair[0]}_vs_{self.camera_pair[1]}"],
        )


class CameraSyncChecker:
    """Verifies temporal synchronization and stream identity across multiple cameras."""

    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize checker with audit configuration.
        
        Args:
            config: Optional audit configuration thresholds.
        """
        self.config = config or AuditConfig()
        self.tolerance_ms = self.config.camera_sync_tolerance_ms  # 16.0ms default
        self.frame_count_tolerance = 2  # 2 frames default

    def check_timestamp_sync(
        self,
        camera_streams: Dict[str, List[Frame]],
        tolerance_ms: Optional[float] = None,
    ) -> List[CameraSyncIssue]:
        """Verify frame timestamps stay synchronized across all camera pairs.
        
        Requirements 10.1, 10.2: Flag camera_desync warning when timestamps
        diverge by more than 16 milliseconds.
        
        Args:
            camera_streams: Mapping from camera ID to list of Frame objects.
            tolerance_ms: Sync tolerance in ms (defaults to 16.0ms).
            
        Returns:
            List of detected CameraSyncIssue instances.
        """
        tol_ms = tolerance_ms if tolerance_ms is not None else self.tolerance_ms
        issues: List[CameraSyncIssue] = []

        cams = sorted(list(camera_streams.keys()))
        if len(cams) < 2:
            return issues

        # Check all unique pairs of cameras
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                cam_a, cam_b = cams[i], cams[j]
                frames_a = camera_streams[cam_a]
                frames_b = camera_streams[cam_b]

                if not frames_a or not frames_b:
                    continue

                min_len = min(len(frames_a), len(frames_b))
                desync_spans: List[Tuple[float, float, float]] = []  # (t_start, t_end, max_diff_ms)

                span_start: Optional[float] = None
                span_end: Optional[float] = None
                span_max_diff = 0.0

                for k in range(min_len):
                    ts_a_ms = frames_a[k].timestamp_us / 1000.0
                    ts_b_ms = frames_b[k].timestamp_us / 1000.0
                    diff_ms = abs(ts_a_ms - ts_b_ms)
                    t_curr_s = (ts_a_ms + ts_b_ms) / 2000.0

                    if diff_ms > tol_ms:
                        if span_start is None:
                            span_start = t_curr_s
                        span_end = t_curr_s
                        span_max_diff = max(span_max_diff, diff_ms)
                    else:
                        if span_start is not None and span_end is not None:
                            desync_spans.append((span_start, span_end, span_max_diff))
                            span_start = None
                            span_end = None
                            span_max_diff = 0.0

                if span_start is not None and span_end is not None:
                    desync_spans.append((span_start, span_end, span_max_diff))

                for t_s, t_e, max_d in desync_spans:
                    issues.append(
                        CameraSyncIssue(
                            issue_type="desync",
                            severity="warning",
                            camera_pair=(cam_a, cam_b),
                            time_range=(round(t_s, 3), round(t_e, 3)),
                            details=(
                                f"Frame timestamp desync between '{cam_a}' and '{cam_b}' "
                                f"reached {max_d:.1f}ms (threshold: {tol_ms}ms) "
                                f"in range [{t_s:.2f}s, {t_e:.2f}s]"
                            ),
                        )
                    )

        return issues

    def check_frame_counts(
        self,
        camera_streams: Dict[str, List[Frame]],
        tolerance_frames: Optional[int] = None,
    ) -> List[CameraSyncIssue]:
        """Verify all camera streams have consistent frame counts.
        
        Requirement 10.3: Flag camera_desync warning when counts differ by > 2 frames.
        
        Args:
            camera_streams: Mapping from camera ID to list of Frame objects.
            tolerance_frames: Frame count tolerance (defaults to 2).
            
        Returns:
            List of detected CameraSyncIssue instances.
        """
        tol = tolerance_frames if tolerance_frames is not None else self.frame_count_tolerance
        issues: List[CameraSyncIssue] = []

        cams = sorted(list(camera_streams.keys()))
        if len(cams) < 2:
            return issues

        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                cam_a, cam_b = cams[i], cams[j]
                count_a = len(camera_streams[cam_a])
                count_b = len(camera_streams[cam_b])
                diff = abs(count_a - count_b)

                if diff > tol:
                    issues.append(
                        CameraSyncIssue(
                            issue_type="frame_count_mismatch",
                            severity="warning",
                            camera_pair=(cam_a, cam_b),
                            time_range=None,
                            details=(
                                f"Frame count discrepancy between '{cam_a}' ({count_a} frames) "
                                f"and '{cam_b}' ({count_b} frames) is {diff} frames "
                                f"(tolerance: {tol} frames)"
                            ),
                        )
                    )

        return issues

    def detect_camera_swap(
        self,
        camera_streams: Dict[str, List[Frame]],
    ) -> List[CameraSyncIssue]:
        """Detect camera stream identifier swaps via viewpoint and feature consistency.
        
        Requirements 10.4, 10.5: Flag high-severity camera_swap errors when camera
        identities switch mid-recording.
        
        Args:
            camera_streams: Mapping from camera ID to list of Frame objects.
            
        Returns:
            List of detected CameraSyncIssue instances with severity 'error'.
        """
        issues: List[CameraSyncIssue] = []

        cams = sorted(list(camera_streams.keys()))
        if len(cams) < 2:
            return issues

        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                cam_a, cam_b = cams[i], cams[j]
                frames_a = camera_streams[cam_a]
                frames_b = camera_streams[cam_b]

                if len(frames_a) < 10 or len(frames_b) < 10:
                    continue

                # Check frame image signatures (mean RGB color / brightness)
                features_a = [self._compute_frame_signature(f) for f in frames_a]
                features_b = [self._compute_frame_signature(f) for f in frames_b]

                # Check if signatures invert in second half
                swap_point = self._detect_signature_cross_inversion(features_a, features_b)
                if swap_point is not None:
                    t_swap = frames_a[swap_point].timestamp_us / 1_000_000.0
                    t_end = frames_a[-1].timestamp_us / 1_000_000.0
                    issues.append(
                        CameraSyncIssue(
                            issue_type="camera_swap",
                            severity="error",
                            camera_pair=(cam_a, cam_b),
                            time_range=(round(t_swap, 3), round(t_end, 3)),
                            details=(
                                f"High-severity camera identifier swap detected between "
                                f"'{cam_a}' and '{cam_b}' at t={t_swap:.2f}s"
                            ),
                        )
                    )

        return issues

    def check_all(
        self,
        camera_streams: Dict[str, List[Frame]],
        tolerance_ms: Optional[float] = None,
        tolerance_frames: Optional[int] = None,
    ) -> List[CameraSyncIssue]:
        """Run all camera sync checks and return aggregated issues."""
        issues: List[CameraSyncIssue] = []
        issues.extend(self.check_timestamp_sync(camera_streams, tolerance_ms))
        issues.extend(self.check_frame_counts(camera_streams, tolerance_frames))
        issues.extend(self.detect_camera_swap(camera_streams))
        return issues

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _compute_frame_signature(self, frame: Frame) -> np.ndarray:
        """Compute compact visual feature vector for a frame."""
        img = frame.image
        if img is None or img.size == 0:
            return np.zeros(3, dtype=float)
        # Average channel intensity
        if len(img.shape) == 3:
            return np.mean(img, axis=(0, 1), dtype=float)
        return np.array([float(np.mean(img))] * 3, dtype=float)

    def _detect_signature_cross_inversion(
        self,
        feat_a: List[np.ndarray],
        feat_b: List[np.ndarray],
    ) -> Optional[int]:
        """Detect if feature signatures cross and swap identities mid-stream."""
        n = min(len(feat_a), len(feat_b))
        if n < 10:
            return None

        # Baseline features from first quarter
        base_split = max(2, n // 4)
        base_a = np.mean(feat_a[:base_split], axis=0)
        base_b = np.mean(feat_b[:base_split], axis=0)

        # Baseline distance between the two cameras
        inter_cam_dist = np.linalg.norm(base_a - base_b)
        if inter_cam_dist < 15.0:
            # Cameras have nearly identical visual signatures; cannot reliably detect visual swap
            return None

        # Check each point after base_split
        consecutive_swaps = 0
        swap_start_idx = None

        for k in range(base_split, n):
            curr_a = feat_a[k]
            curr_b = feat_b[k]

            # In normal state: dist(curr_a, base_a) < dist(curr_a, base_b)
            # In swapped state: dist(curr_a, base_b) < dist(curr_a, base_a)
            dist_aa = np.linalg.norm(curr_a - base_a)
            dist_ab = np.linalg.norm(curr_a - base_b)
            dist_ba = np.linalg.norm(curr_b - base_a)
            dist_bb = np.linalg.norm(curr_b - base_b)

            if dist_ab < dist_aa and dist_ba < dist_bb:
                consecutive_swaps += 1
                if consecutive_swaps >= 3:
                    if swap_start_idx is None:
                        swap_start_idx = k - 2
                    return swap_start_idx
            else:
                consecutive_swaps = 0

        return None
