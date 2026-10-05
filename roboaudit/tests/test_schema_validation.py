"""Unit tests for JSON schema validation."""

import pytest
from roboaudit.core.schema import validate_audit_report, SCHEMA_VERSION


class TestSchemaValidation:
    """Tests for audit report schema validation."""
    
    def test_valid_minimal_report(self):
        """Test validation of a minimal valid report."""
        report = {
            "schema_version": SCHEMA_VERSION,
            "context": {
                "dataset": "test_dataset",
                "rig": "rig_01",
                "length_s": 30.0,
                "instruction": "Pick up the cube",
                "episode_id": "ep_001"
            },
            "timeline": [
                {
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "arm": "left",
                    "action": "approach",
                    "object": None,
                    "contribution": "advancing",
                    "progress": 0.3
                }
            ],
            "completion": {
                "task_completed": True,
                "goal_reached_at_s": 25.0,
                "undone_at_s": None,
                "completed_at_s": 25.0,
                "reason": "Successfully completed task"
            },
            "goal_alignment": {
                "matches_given": True,
                "relation": "exact",
                "note": "Perfect alignment"
            },
            "data_issues": [],
            "operator_mistakes": [],
            "quality_metrics": {
                "total_anomalies": 0,
                "critical_issues": 0,
                "warning_count": 0,
                "goal_alignment_score": 1.0,
                "quality_score": 1.0
            }
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is True
        assert error is None
    
    def test_missing_schema_version(self):
        """Test validation fails when schema_version is missing."""
        report = {
            "context": {},
            "timeline": [],
            "completion": {},
            "goal_alignment": {},
            "data_issues": [],
            "operator_mistakes": [],
            "quality_metrics": {}
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is False
        assert "schema_version" in error
    
    def test_unsupported_schema_version(self):
        """Test validation fails with unsupported schema version."""
        report = {
            "schema_version": "2.0.0",
            "context": {},
            "timeline": [],
            "completion": {},
            "goal_alignment": {},
            "data_issues": [],
            "operator_mistakes": [],
            "quality_metrics": {}
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is False
        assert "Unsupported schema version" in error
    
    def test_missing_required_field(self):
        """Test validation fails when required field is missing."""
        report = {
            "schema_version": SCHEMA_VERSION,
            "context": {},
            # Missing timeline
            "completion": {},
            "goal_alignment": {},
            "data_issues": [],
            "operator_mistakes": [],
            "quality_metrics": {}
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is False
        assert "timeline" in error
    
    def test_invalid_context_length(self):
        """Test validation fails with negative episode length."""
        report = {
            "schema_version": SCHEMA_VERSION,
            "context": {
                "dataset": "test",
                "rig": "rig_01",
                "length_s": -10.0,  # Invalid
                "instruction": "Test",
                "episode_id": "ep_001"
            },
            "timeline": [],
            "completion": {"task_completed": True, "reason": "Test"},
            "goal_alignment": {"matches_given": True, "relation": "exact", "note": "Test"},
            "data_issues": [],
            "operator_mistakes": [],
            "quality_metrics": {
                "total_anomalies": 0,
                "critical_issues": 0,
                "warning_count": 0,
                "goal_alignment_score": 0.5,
                "quality_score": 0.5
            }
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is False
        assert "non-negative" in error
    
    def test_invalid_timeline_arm_value(self):
        """Test validation fails with invalid arm value."""
        report = {
            "schema_version": SCHEMA_VERSION,
            "context": {
                "dataset": "test",
                "rig": "rig_01",
                "length_s": 30.0,
                "instruction": "Test",
                "episode_id": "ep_001"
            },
            "timeline": [
                {
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "arm": "invalid_arm",  # Invalid
                    "action": "approach",
                    "contribution": "advancing",
                    "progress": 0.3
                }
            ],
            "completion": {"task_completed": True, "reason": "Test"},
            "goal_alignment": {"matches_given": True, "relation": "exact", "note": "Test"},
            "data_issues": [],
            "operator_mistakes": [],
            "quality_metrics": {
                "total_anomalies": 0,
                "critical_issues": 0,
                "warning_count": 0,
                "goal_alignment_score": 0.5,
                "quality_score": 0.5
            }
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is False
        assert "arm" in error
    
    def test_invalid_progress_range(self):
        """Test validation fails with progress outside [0.0, 1.0]."""
        report = {
            "schema_version": SCHEMA_VERSION,
            "context": {
                "dataset": "test",
                "rig": "rig_01",
                "length_s": 30.0,
                "instruction": "Test",
                "episode_id": "ep_001"
            },
            "timeline": [
                {
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "arm": "left",
                    "action": "approach",
                    "contribution": "advancing",
                    "progress": 1.5  # Invalid
                }
            ],
            "completion": {"task_completed": True, "reason": "Test"},
            "goal_alignment": {"matches_given": True, "relation": "exact", "note": "Test"},
            "data_issues": [],
            "operator_mistakes": [],
            "quality_metrics": {
                "total_anomalies": 0,
                "critical_issues": 0,
                "warning_count": 0,
                "goal_alignment_score": 0.5,
                "quality_score": 0.5
            }
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is False
        assert "progress" in error
    
    def test_invalid_data_issue_category(self):
        """Test validation fails with invalid data issue category."""
        report = {
            "schema_version": SCHEMA_VERSION,
            "context": {
                "dataset": "test",
                "rig": "rig_01",
                "length_s": 30.0,
                "instruction": "Test",
                "episode_id": "ep_001"
            },
            "timeline": [],
            "completion": {"task_completed": True, "reason": "Test"},
            "goal_alignment": {"matches_given": True, "relation": "exact", "note": "Test"},
            "data_issues": [
                {
                    "issue": "Test issue",
                    "category": "invalid_category",  # Invalid
                    "severity": "low",
                    "t_s": 5.0,
                    "evidence": []
                }
            ],
            "operator_mistakes": [],
            "quality_metrics": {
                "total_anomalies": 1,
                "critical_issues": 0,
                "warning_count": 1,
                "goal_alignment_score": 0.5,
                "quality_score": 0.5
            }
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is False
        assert "category" in error
    
    def test_invalid_quality_score_range(self):
        """Test validation fails with quality score outside [0.0, 1.0]."""
        report = {
            "schema_version": SCHEMA_VERSION,
            "context": {
                "dataset": "test",
                "rig": "rig_01",
                "length_s": 30.0,
                "instruction": "Test",
                "episode_id": "ep_001"
            },
            "timeline": [],
            "completion": {"task_completed": True, "reason": "Test"},
            "goal_alignment": {"matches_given": True, "relation": "exact", "note": "Test"},
            "data_issues": [],
            "operator_mistakes": [],
            "quality_metrics": {
                "total_anomalies": 0,
                "critical_issues": 0,
                "warning_count": 0,
                "goal_alignment_score": 1.5,  # Invalid
                "quality_score": 0.5
            }
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is False
        assert "goal_alignment_score" in error
    
    def test_valid_report_with_issues(self):
        """Test validation of report with data issues and operator mistakes."""
        report = {
            "schema_version": SCHEMA_VERSION,
            "context": {
                "dataset": "test_dataset",
                "rig": "rig_01",
                "length_s": 30.0,
                "instruction": "Pick up the cube",
                "episode_id": "ep_001"
            },
            "timeline": [
                {
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "arm": "left",
                    "action": "approach",
                    "object": "cube",
                    "contribution": "advancing",
                    "progress": 0.3
                }
            ],
            "completion": {
                "task_completed": False,
                "goal_reached_at_s": None,
                "undone_at_s": None,
                "completed_at_s": None,
                "reason": "Object dropped"
            },
            "goal_alignment": {
                "matches_given": True,
                "relation": "exact",
                "note": "Attempted correctly"
            },
            "data_issues": [
                {
                    "issue": "Frame jitter detected",
                    "category": "timebase",
                    "severity": "low",
                    "t_s": 5.5,
                    "evidence": ["frame_150"]
                },
                {
                    "issue": "Phantom grasp detected",
                    "category": "grasp",
                    "severity": "high",
                    "t_s": 12.0,
                    "evidence": ["frame_360", "cam_wrist"]
                }
            ],
            "operator_mistakes": [
                {
                    "type": "drop",
                    "severity": "high",
                    "t_s": 12.0,
                    "duration_s": None,
                    "evidence": ["frame_360"]
                }
            ],
            "quality_metrics": {
                "total_anomalies": 3,
                "critical_issues": 2,
                "warning_count": 1,
                "goal_alignment_score": 0.4,
                "quality_score": 0.35
            }
        }
        
        is_valid, error = validate_audit_report(report)
        assert is_valid is True
        assert error is None
