"""Core data models for RoboAudit."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class Frame:
    """A single video frame with timestamp and metadata.
    
    Attributes:
        timestamp_us: Microsecond precision timestamp
        pts: Presentation timestamp from video container
        image: Frame image in BGR format (numpy array)
        camera_id: Camera identifier string
    """
    timestamp_us: int
    pts: int
    image: np.ndarray
    camera_id: str


@dataclass
class EpisodeMetadata:
    """Metadata describing a single episode.
    
    Attributes:
        episode_id: Unique episode identifier
        duration_s: Episode duration in seconds
        recording_date: Optional timestamp of when episode was recorded
        robot_type: Robot platform identifier (e.g., "umi", "franka", "kinova")
        camera_count: Number of video streams in episode
        telemetry_channels: List of telemetry channel names
    """
    episode_id: str
    duration_s: float
    recording_date: Optional[datetime]
    robot_type: str
    camera_count: int
    telemetry_channels: List[str]


@dataclass
class ExtractedEpisode:
    """Complete extracted episode data.
    
    Attributes:
        metadata: Episode metadata
        video_streams: Dictionary mapping camera IDs to frame lists
        telemetry: DataFrame with timestamp column and telemetry channels
    """
    metadata: EpisodeMetadata
    video_streams: Dict[str, List[Frame]]
    telemetry: pd.DataFrame


@dataclass
class TemporalWindow:
    """A temporal window in the episode timeline with action classification.
    
    Attributes:
        start_s: Window start time in seconds
        end_s: Window end time in seconds
        action_phase: Action phase classification
        arm_attribution: Which arm(s) are active
        contribution_type: Whether window advances task, wastes time, or idles
        completion_percentage: Task completion progress (0.0 to 1.0)
        object: Optional object being manipulated
    """
    start_s: float
    end_s: float
    action_phase: Literal["approach", "grasp", "manipulate", "release", "idle"]
    arm_attribution: Literal["left", "right", "both", "none"]
    contribution_type: Literal["advancing", "wasteful", "idle"]
    completion_percentage: float
    object: Optional[str] = None
    
    def __post_init__(self):
        """Validate temporal window constraints."""
        if self.start_s < 0:
            raise ValueError("start_s must be non-negative")
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        if not 0.0 <= self.completion_percentage <= 1.0:
            raise ValueError("completion_percentage must be in [0.0, 1.0]")


@dataclass
class Timeline:
    """Sequence of temporal windows representing episode timeline.
    
    Attributes:
        windows: List of temporal windows in chronological order
    """
    windows: List[TemporalWindow]
    
    def validate_monotonicity(self) -> bool:
        """Verify completion is non-decreasing during advancing phases.
        
        Returns:
            True if monotonicity holds, False otherwise
        """
        for i in range(len(self.windows) - 1):
            current = self.windows[i]
            next_window = self.windows[i + 1]
            
            # During advancing phases, completion should not decrease
            if current.contribution_type == "advancing":
                if next_window.contribution_type == "advancing":
                    if next_window.completion_percentage < current.completion_percentage:
                        return False
        
        return True


@dataclass
class TaskOutcome:
    """Task outcome classification with temporal markers.
    
    Attributes:
        outcome: Final outcome classification
        goal_reached_at_s: Optional timestamp when goal was reached
        undone_at_s: Optional timestamp when task was undone
        completed_at_s: Optional timestamp when task was completed
    """
    outcome: Literal["success", "failure", "partial", "success_then_undone"]
    goal_reached_at_s: Optional[float] = None
    undone_at_s: Optional[float] = None
    completed_at_s: Optional[float] = None


@dataclass
class AuditConfig:
    """Configuration for audit thresholds and parameters.
    
    Attributes:
        jitter_threshold: Acceptable frame/telemetry jitter as percentage (default 2.0%)
        sensor_desync_tolerance_ms: Maximum acceptable video-telemetry lag in milliseconds
        camera_sync_tolerance_ms: Maximum acceptable multi-camera desync in milliseconds
        sensor_gap_threshold_ms: Maximum acceptable telemetry gap in milliseconds
        hesitation_threshold_s: Minimum idle duration to flag as hesitation
        alignment_angle_threshold_deg: Maximum alignment error before flagging struggle
        robot_type: Robot platform identifier
        joint_limits: Optional joint angle limits for outlier detection
    """
    jitter_threshold: float = 0.02  # 2.0%
    sensor_desync_tolerance_ms: float = 50.0
    camera_sync_tolerance_ms: float = 16.0
    sensor_gap_threshold_ms: float = 100.0
    hesitation_threshold_s: float = 2.0
    alignment_angle_threshold_deg: float = 15.0
    robot_type: str = "generic"
    joint_limits: Optional[Dict[str, Tuple[float, float]]] = None


@dataclass
class EpisodeContext:
    """Context information for an audited episode.
    
    Attributes:
        dataset: Dataset name
        rig: Recording rig identifier
        length_s: Episode duration in seconds
        instruction: Task instruction text
        episode_id: Episode identifier
    """
    dataset: str
    rig: str
    length_s: float
    instruction: str
    episode_id: str


@dataclass
class TaskCompletion:
    """Task completion status and temporal markers.
    
    Attributes:
        task_completed: Whether task was successfully completed
        goal_reached_at_s: Optional timestamp when goal was reached
        undone_at_s: Optional timestamp when task was undone
        completed_at_s: Optional timestamp when task was fully completed
        reason: Human-readable reason for completion status
    """
    task_completed: bool
    goal_reached_at_s: Optional[float]
    undone_at_s: Optional[float]
    completed_at_s: Optional[float]
    reason: str


@dataclass
class GoalAlignment:
    """Goal alignment assessment.
    
    Attributes:
        matches_given: Whether episode matches given instruction
        relation: Relationship to instruction (e.g., "exact", "partial", "contradicts")
        note: Additional notes on alignment
    """
    matches_given: bool
    relation: str
    note: str


@dataclass
class DataIssue:
    """A detected data quality issue.
    
    Attributes:
        issue: Issue description
        category: Issue category
        severity: Severity level
        t_s: Timestamp in seconds
        evidence: List of evidence references (e.g., frame numbers)
    """
    issue: str
    category: Literal["timebase", "sensor", "camera", "grasp"]
    severity: Literal["low", "medium", "high", "error"]
    t_s: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Aggregate quality metrics for an episode.
    
    Attributes:
        total_anomalies: Total number of anomalies detected
        critical_issues: Number of critical/high-severity issues
        warning_count: Number of warnings
        goal_alignment_score: Goal alignment score (0.0 to 1.0)
        quality_score: Overall quality score (0.0 to 1.0)
    """
    total_anomalies: int
    critical_issues: int
    warning_count: int
    goal_alignment_score: float
    quality_score: float
    
    def __post_init__(self):
        """Validate metric ranges."""
        if not 0.0 <= self.goal_alignment_score <= 1.0:
            raise ValueError("goal_alignment_score must be in [0.0, 1.0]")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be in [0.0, 1.0]")


@dataclass
class OperatorMistake:
    """An operator execution mistake detected during demonstration."""
    type: Literal["drop", "alignment_struggle", "hesitation", "collision", "fumble"]
    severity: Literal["low", "medium", "high", "error"]
    t_s: float
    duration_s: Optional[float] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Complete structured audit report for a robotics demonstration episode."""
    schema_version: str
    context: EpisodeContext
    timeline: Timeline
    completion: TaskCompletion
    goal_alignment: GoalAlignment
    data_issues: List[DataIssue]
    operator_mistakes: List[OperatorMistake]
    quality_metrics: QualityMetrics
