"""Unit tests for core data models."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from hypothesis import given, settings

from roboaudit.tests import strategies as rst

from roboaudit.core import (
    EpisodeMetadata,
    ExtractedEpisode,
    Frame,
    TemporalWindow,
    Timeline,
    TaskOutcome,
    AuditConfig,
    EpisodeContext,
    TaskCompletion,
    GoalAlignment,
    DataIssue,
    QualityMetrics,
)


class TestFrame:
    """Tests for Frame data class."""
    
    def test_frame_creation(self):
        """Test creating a valid frame."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            timestamp_us=1000000,
            pts=30,
            image=image,
            camera_id="wrist_camera"
        )
        
        assert frame.timestamp_us == 1000000
        assert frame.pts == 30
        assert frame.image.shape == (480, 640, 3)
        assert frame.camera_id == "wrist_camera"


class TestEpisodeMetadata:
    """Tests for EpisodeMetadata data class."""
    
    def test_episode_metadata_creation(self):
        """Test creating episode metadata."""
        metadata = EpisodeMetadata(
            episode_id="ep_001",
            duration_s=45.5,
            recording_date=datetime(2024, 1, 15, 10, 30),
            robot_type="umi",
            camera_count=2,
            telemetry_channels=["joint_pos", "gripper_state", "force"]
        )
        
        assert metadata.episode_id == "ep_001"
        assert metadata.duration_s == 45.5
        assert metadata.robot_type == "umi"
        assert metadata.camera_count == 2
        assert len(metadata.telemetry_channels) == 3
    
    def test_episode_metadata_optional_date(self):
        """Test creating episode metadata without recording date."""
        metadata = EpisodeMetadata(
            episode_id="ep_002",
            duration_s=30.0,
            recording_date=None,
            robot_type="franka",
            camera_count=1,
            telemetry_channels=["joint_pos"]
        )
        
        assert metadata.recording_date is None


class TestExtractedEpisode:
    """Tests for ExtractedEpisode data class."""
    
    def test_extracted_episode_creation(self):
        """Test creating extracted episode."""
        metadata = EpisodeMetadata(
            episode_id="ep_001",
            duration_s=10.0,
            recording_date=None,
            robot_type="generic",
            camera_count=1,
            telemetry_channels=["joint_pos"]
        )
        
        frame = Frame(
            timestamp_us=1000000,
            pts=30,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            camera_id="cam1"
        )
        
        telemetry = pd.DataFrame({
            "timestamp": [0.0, 0.1, 0.2],
            "joint_pos": [0.0, 0.5, 1.0]
        })
        
        episode = ExtractedEpisode(
            metadata=metadata,
            video_streams={"cam1": [frame]},
            telemetry=telemetry
        )
        
        assert episode.metadata.episode_id == "ep_001"
        assert len(episode.video_streams) == 1
        assert len(episode.telemetry) == 3


class TestTemporalWindow:
    """Tests for TemporalWindow data class."""
    
    def test_temporal_window_creation(self):
        """Test creating a valid temporal window."""
        window = TemporalWindow(
            start_s=0.0,
            end_s=5.0,
            action_phase="approach",
            arm_attribution="left",
            contribution_type="advancing",
            completion_percentage=0.2
        )
        
        assert window.start_s == 0.0
        assert window.end_s == 5.0
        assert window.action_phase == "approach"
        assert window.completion_percentage == 0.2
    
    def test_temporal_window_invalid_time_range(self):
        """Test that invalid time ranges raise ValueError."""
        with pytest.raises(ValueError, match="end_s must be greater than start_s"):
            TemporalWindow(
                start_s=5.0,
                end_s=5.0,
                action_phase="idle",
                arm_attribution="none",
                contribution_type="idle",
                completion_percentage=0.5
            )
    
    def test_temporal_window_negative_start(self):
        """Test that negative start time raises ValueError."""
        with pytest.raises(ValueError, match="start_s must be non-negative"):
            TemporalWindow(
                start_s=-1.0,
                end_s=5.0,
                action_phase="idle",
                arm_attribution="none",
                contribution_type="idle",
                completion_percentage=0.5
            )
    
    def test_temporal_window_invalid_completion(self):
        """Test that invalid completion percentage raises ValueError."""
        with pytest.raises(ValueError, match="completion_percentage must be in"):
            TemporalWindow(
                start_s=0.0,
                end_s=5.0,
                action_phase="manipulate",
                arm_attribution="both",
                contribution_type="advancing",
                completion_percentage=1.5
            )


class TestTimeline:
    """Tests for Timeline data class."""
    
    def test_timeline_creation(self):
        """Test creating a timeline."""
        windows = [
            TemporalWindow(0.0, 5.0, "approach", "left", "advancing", 0.2),
            TemporalWindow(5.0, 10.0, "grasp", "left", "advancing", 0.5),
        ]
        
        timeline = Timeline(windows=windows)
        assert len(timeline.windows) == 2
    
    def test_timeline_validate_monotonicity_valid(self):
        """Test monotonicity validation with valid progression."""
        windows = [
            TemporalWindow(0.0, 5.0, "approach", "left", "advancing", 0.2),
            TemporalWindow(5.0, 10.0, "grasp", "left", "advancing", 0.5),
            TemporalWindow(10.0, 15.0, "manipulate", "left", "advancing", 0.8),
        ]
        
        timeline = Timeline(windows=windows)
        assert timeline.validate_monotonicity() is True
    
    def test_timeline_validate_monotonicity_invalid(self):
        """Test monotonicity validation with decreasing completion."""
        windows = [
            TemporalWindow(0.0, 5.0, "approach", "left", "advancing", 0.5),
            TemporalWindow(5.0, 10.0, "grasp", "left", "advancing", 0.3),
        ]
        
        timeline = Timeline(windows=windows)
        assert timeline.validate_monotonicity() is False
    
    def test_timeline_validate_monotonicity_with_idle(self):
        """Test monotonicity validation allows idle periods."""
        windows = [
            TemporalWindow(0.0, 5.0, "approach", "left", "advancing", 0.5),
            TemporalWindow(5.0, 10.0, "idle", "none", "idle", 0.5),
            TemporalWindow(10.0, 15.0, "manipulate", "left", "advancing", 0.8),
        ]
        
        timeline = Timeline(windows=windows)
        assert timeline.validate_monotonicity() is True

    @settings(max_examples=100)
    @given(timeline=rst.timelines(monotonic=True))
    def test_property_monotonic_timelines_validate(self, timeline):
        """Property: any timeline from the shared monotonic strategy satisfies
        validate_monotonicity(). Demonstrates reuse of the shared Lesson 4
        strategies module instead of hand-written example windows."""
        assert timeline.validate_monotonicity() is True
        # Windows are chronologically ordered (end of one <= start of next... actually == start).
        for prev, nxt in zip(timeline.windows, timeline.windows[1:]):
            assert nxt.start_s >= prev.start_s


class TestTaskOutcome:
    """Tests for TaskOutcome data class."""
    
    def test_task_outcome_success(self):
        """Test success outcome."""
        outcome = TaskOutcome(
            outcome="success",
            goal_reached_at_s=10.0,
            completed_at_s=10.5
        )
        
        assert outcome.outcome == "success"
        assert outcome.goal_reached_at_s == 10.0
    
    def test_task_outcome_success_then_undone(self):
        """Test success_then_undone outcome."""
        outcome = TaskOutcome(
            outcome="success_then_undone",
            goal_reached_at_s=10.0,
            undone_at_s=15.0
        )
        
        assert outcome.outcome == "success_then_undone"
        assert outcome.undone_at_s == 15.0


class TestAuditConfig:
    """Tests for AuditConfig data class."""
    
    def test_audit_config_defaults(self):
        """Test creating config with default values."""
        config = AuditConfig()
        
        assert config.jitter_threshold == 0.02
        assert config.sensor_desync_tolerance_ms == 50.0
        assert config.camera_sync_tolerance_ms == 16.0
        assert config.sensor_gap_threshold_ms == 100.0
        assert config.hesitation_threshold_s == 2.0
        assert config.alignment_angle_threshold_deg == 15.0
        assert config.robot_type == "generic"
        assert config.joint_limits is None
    
    def test_audit_config_custom_values(self):
        """Test creating config with custom values."""
        config = AuditConfig(
            jitter_threshold=0.03,
            robot_type="umi",
            joint_limits={"joint1": (-180.0, 180.0)}
        )
        
        assert config.jitter_threshold == 0.03
        assert config.robot_type == "umi"
        assert config.joint_limits is not None


class TestDataIssue:
    """Tests for DataIssue data class."""
    
    def test_data_issue_creation(self):
        """Test creating a data issue."""
        issue = DataIssue(
            issue="Frame jitter detected",
            category="timebase",
            severity="warning",
            t_s=5.5,
            evidence=["frame_150", "frame_151"]
        )
        
        assert issue.issue == "Frame jitter detected"
        assert issue.category == "timebase"
        assert issue.severity == "warning"
        assert len(issue.evidence) == 2


class TestQualityMetrics:
    """Tests for QualityMetrics data class."""
    
    def test_quality_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            total_anomalies=5,
            critical_issues=1,
            warning_count=4,
            goal_alignment_score=0.85,
            quality_score=0.92
        )
        
        assert metrics.total_anomalies == 5
        assert metrics.critical_issues == 1
        assert metrics.goal_alignment_score == 0.85
    
    def test_quality_metrics_invalid_scores(self):
        """Test that invalid scores raise ValueError."""
        with pytest.raises(ValueError, match="goal_alignment_score must be in"):
            QualityMetrics(
                total_anomalies=0,
                critical_issues=0,
                warning_count=0,
                goal_alignment_score=1.5,
                quality_score=0.5
            )
