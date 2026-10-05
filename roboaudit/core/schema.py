"""JSON schema validation for audit reports."""

from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0.0"

# JSON schema for audit reports
AUDIT_REPORT_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version",
        "context",
        "timeline",
        "completion",
        "goal_alignment",
        "data_issues",
        "operator_mistakes",
        "quality_metrics"
    ],
    "properties": {
        "schema_version": {
            "type": "string",
            "const": SCHEMA_VERSION
        },
        "context": {
            "type": "object",
            "required": ["dataset", "rig", "length_s", "instruction", "episode_id"],
            "properties": {
                "dataset": {"type": "string"},
                "rig": {"type": "string"},
                "length_s": {"type": "number", "minimum": 0},
                "instruction": {"type": "string"},
                "episode_id": {"type": "string"}
            }
        },
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["start_s", "end_s", "arm", "action", "contribution", "progress"],
                "properties": {
                    "start_s": {"type": "number", "minimum": 0},
                    "end_s": {"type": "number", "minimum": 0},
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right", "both", "none"]
                    },
                    "action": {
                        "type": "string",
                        "enum": ["approach", "grasp", "manipulate", "release", "idle"]
                    },
                    "object": {"type": ["string", "null"]},
                    "contribution": {
                        "type": "string",
                        "enum": ["advancing", "wasteful", "idle"]
                    },
                    "progress": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            }
        },
        "completion": {
            "type": "object",
            "required": ["task_completed", "reason"],
            "properties": {
                "task_completed": {"type": "boolean"},
                "goal_reached_at_s": {"type": ["number", "null"]},
                "undone_at_s": {"type": ["number", "null"]},
                "completed_at_s": {"type": ["number", "null"]},
                "reason": {"type": "string"}
            }
        },
        "goal_alignment": {
            "type": "object",
            "required": ["matches_given", "relation", "note"],
            "properties": {
                "matches_given": {"type": "boolean"},
                "relation": {"type": "string"},
                "note": {"type": "string"}
            }
        },
        "data_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["issue", "category", "severity", "t_s", "evidence"],
                "properties": {
                    "issue": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["timebase", "sensor", "camera", "grasp"]
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "error"]
                    },
                    "t_s": {"type": "number"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        },
        "operator_mistakes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "severity", "t_s", "evidence"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["drop", "alignment_struggle", "hesitation", "fumble"]
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"]
                    },
                    "t_s": {"type": "number"},
                    "duration_s": {"type": ["number", "null"]},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        },
        "quality_metrics": {
            "type": "object",
            "required": [
                "total_anomalies",
                "critical_issues",
                "warning_count",
                "goal_alignment_score",
                "quality_score"
            ],
            "properties": {
                "total_anomalies": {"type": "integer", "minimum": 0},
                "critical_issues": {"type": "integer", "minimum": 0},
                "warning_count": {"type": "integer", "minimum": 0},
                "goal_alignment_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "quality_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            }
        }
    }
}


def validate_audit_report(report_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate an audit report dictionary against the schema.
    
    Args:
        report_dict: Dictionary representation of an audit report
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, returns (True, None)
        If invalid, returns (False, error_message)
    """
    # Check schema version
    if "schema_version" not in report_dict:
        return False, "Missing required field: schema_version"
    
    if report_dict["schema_version"] != SCHEMA_VERSION:
        return False, f"Unsupported schema version: {report_dict['schema_version']}"
    
    # Check required top-level fields
    required_fields = [
        "schema_version", "context", "timeline", "completion",
        "goal_alignment", "data_issues", "operator_mistakes", "quality_metrics"
    ]
    
    for field in required_fields:
        if field not in report_dict:
            return False, f"Missing required field: {field}"
    
    # Validate context
    context_required = ["dataset", "rig", "length_s", "instruction", "episode_id"]
    for field in context_required:
        if field not in report_dict["context"]:
            return False, f"Missing required context field: {field}"
    
    if not isinstance(report_dict["context"]["length_s"], (int, float)):
        return False, "context.length_s must be a number"
    
    if report_dict["context"]["length_s"] < 0:
        return False, "context.length_s must be non-negative"
    
    # Validate timeline
    if not isinstance(report_dict["timeline"], list):
        return False, "timeline must be an array"
    
    for i, window in enumerate(report_dict["timeline"]):
        window_required = ["start_s", "end_s", "arm", "action", "contribution", "progress"]
        for field in window_required:
            if field not in window:
                return False, f"timeline[{i}] missing required field: {field}"
        
        if window["arm"] not in ["left", "right", "both", "none"]:
            return False, f"timeline[{i}].arm has invalid value: {window['arm']}"
        
        if window["action"] not in ["approach", "grasp", "manipulate", "release", "idle"]:
            return False, f"timeline[{i}].action has invalid value: {window['action']}"
        
        if window["contribution"] not in ["advancing", "wasteful", "idle"]:
            return False, f"timeline[{i}].contribution has invalid value: {window['contribution']}"
        
        if not isinstance(window["progress"], (int, float)):
            return False, f"timeline[{i}].progress must be a number"
        
        if not 0.0 <= window["progress"] <= 1.0:
            return False, f"timeline[{i}].progress must be in [0.0, 1.0]"
    
    # Validate completion
    completion_required = ["task_completed", "reason"]
    for field in completion_required:
        if field not in report_dict["completion"]:
            return False, f"Missing required completion field: {field}"
    
    if not isinstance(report_dict["completion"]["task_completed"], bool):
        return False, "completion.task_completed must be a boolean"
    
    # Validate goal_alignment
    goal_required = ["matches_given", "relation", "note"]
    for field in goal_required:
        if field not in report_dict["goal_alignment"]:
            return False, f"Missing required goal_alignment field: {field}"
    
    if not isinstance(report_dict["goal_alignment"]["matches_given"], bool):
        return False, "goal_alignment.matches_given must be a boolean"
    
    # Validate data_issues
    if not isinstance(report_dict["data_issues"], list):
        return False, "data_issues must be an array"
    
    for i, issue in enumerate(report_dict["data_issues"]):
        issue_required = ["issue", "category", "severity", "t_s", "evidence"]
        for field in issue_required:
            if field not in issue:
                return False, f"data_issues[{i}] missing required field: {field}"
        
        if issue["category"] not in ["timebase", "sensor", "camera", "grasp"]:
            return False, f"data_issues[{i}].category has invalid value: {issue['category']}"
        
        if issue["severity"] not in ["low", "medium", "high", "error"]:
            return False, f"data_issues[{i}].severity has invalid value: {issue['severity']}"
        
        if not isinstance(issue["evidence"], list):
            return False, f"data_issues[{i}].evidence must be an array"
    
    # Validate operator_mistakes
    if not isinstance(report_dict["operator_mistakes"], list):
        return False, "operator_mistakes must be an array"
    
    for i, mistake in enumerate(report_dict["operator_mistakes"]):
        mistake_required = ["type", "severity", "t_s", "evidence"]
        for field in mistake_required:
            if field not in mistake:
                return False, f"operator_mistakes[{i}] missing required field: {field}"
        
        if mistake["type"] not in ["drop", "alignment_struggle", "hesitation", "fumble"]:
            return False, f"operator_mistakes[{i}].type has invalid value: {mistake['type']}"
        
        if mistake["severity"] not in ["low", "medium", "high"]:
            return False, f"operator_mistakes[{i}].severity has invalid value: {mistake['severity']}"
        
        if not isinstance(mistake["evidence"], list):
            return False, f"operator_mistakes[{i}].evidence must be an array"
    
    # Validate quality_metrics
    metrics_required = [
        "total_anomalies", "critical_issues", "warning_count",
        "goal_alignment_score", "quality_score"
    ]
    for field in metrics_required:
        if field not in report_dict["quality_metrics"]:
            return False, f"Missing required quality_metrics field: {field}"
    
    metrics = report_dict["quality_metrics"]
    
    for field in ["total_anomalies", "critical_issues", "warning_count"]:
        if not isinstance(metrics[field], int):
            return False, f"quality_metrics.{field} must be an integer"
        if metrics[field] < 0:
            return False, f"quality_metrics.{field} must be non-negative"
    
    for field in ["goal_alignment_score", "quality_score"]:
        if not isinstance(metrics[field], (int, float)):
            return False, f"quality_metrics.{field} must be a number"
        if not 0.0 <= metrics[field] <= 1.0:
            return False, f"quality_metrics.{field} must be in [0.0, 1.0]"
    
    return True, None
