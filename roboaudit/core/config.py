"""Configuration loader, validator, and robot profiles for the RoboAudit engine.

Supports:
- Loading configuration from JSON, YAML, or dictionary (Req 12.1)
- Documented default thresholds when none provided (Req 12.2)
- Strict validation with descriptive error reporting (Req 12.3)
- Active configuration export for audit report metadata (Req 12.4)
- Preset configuration profiles for robot platforms (umi, franka, kinova) (Req 12.5)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from roboaudit.core.models import AuditConfig


# Platform-specific profiles (Requirement 12.5)
ROBOT_PROFILES: Dict[str, Dict[str, Any]] = {
    "generic": {
        "jitter_threshold": 0.02,
        "sensor_desync_tolerance_ms": 50.0,
        "camera_sync_tolerance_ms": 16.0,
        "sensor_gap_threshold_ms": 100.0,
        "hesitation_threshold_s": 2.0,
        "alignment_angle_threshold_deg": 15.0,
        "robot_type": "generic",
        "joint_limits": None,
    },
    "umi": {
        "jitter_threshold": 0.015,
        "sensor_desync_tolerance_ms": 33.0,
        "camera_sync_tolerance_ms": 16.0,
        "sensor_gap_threshold_ms": 66.0,
        "hesitation_threshold_s": 1.5,
        "alignment_angle_threshold_deg": 12.0,
        "robot_type": "umi",
        "joint_limits": {
            "gripper_pos": (0.0, 1.0),
        },
    },
    "franka": {
        "jitter_threshold": 0.02,
        "sensor_desync_tolerance_ms": 50.0,
        "camera_sync_tolerance_ms": 16.0,
        "sensor_gap_threshold_ms": 100.0,
        "hesitation_threshold_s": 2.0,
        "alignment_angle_threshold_deg": 15.0,
        "robot_type": "franka",
        "joint_limits": {
            "joint_1": (-2.8973, 2.8973),
            "joint_2": (-1.7628, 1.7628),
            "joint_3": (-2.8973, 2.8973),
            "joint_4": (-3.0718, -0.0698),
            "joint_5": (-2.8973, 2.8973),
            "joint_6": (-0.0175, 3.7525),
            "joint_7": (-2.8973, 2.8973),
        },
    },
    "kinova": {
        "jitter_threshold": 0.02,
        "sensor_desync_tolerance_ms": 50.0,
        "camera_sync_tolerance_ms": 16.0,
        "sensor_gap_threshold_ms": 100.0,
        "hesitation_threshold_s": 2.0,
        "alignment_angle_threshold_deg": 15.0,
        "robot_type": "kinova",
        "joint_limits": {
            "joint_1": (-3.1415, 3.1415),
            "joint_2": (-2.41, 2.41),
            "joint_3": (-3.1415, 3.1415),
            "joint_4": (-2.66, 2.66),
            "joint_5": (-3.1415, 3.1415),
            "joint_6": (-2.23, 2.23),
            "joint_7": (-3.1415, 3.1415),
        },
    },
}


def validate_config_dict(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate configuration settings and return validation result with error description.
    
    Requirement 12.3: Report descriptive errors for invalid configuration settings.
    
    Args:
        data: Dictionary of configuration settings.
        
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str]).
    """
    if not isinstance(data, dict):
        return False, "Configuration must be a dictionary or mapping"

    if "jitter_threshold" in data:
        jt = data["jitter_threshold"]
        if not isinstance(jt, (int, float)) or jt <= 0.0 or jt > 1.0:
            return False, f"jitter_threshold must be a positive float in (0.0, 1.0], got {jt}"

    if "sensor_desync_tolerance_ms" in data:
        desync = data["sensor_desync_tolerance_ms"]
        if not isinstance(desync, (int, float)) or desync <= 0:
            return False, f"sensor_desync_tolerance_ms must be positive, got {desync}"

    if "camera_sync_tolerance_ms" in data:
        csync = data["camera_sync_tolerance_ms"]
        if not isinstance(csync, (int, float)) or csync <= 0:
            return False, f"camera_sync_tolerance_ms must be positive, got {csync}"

    if "sensor_gap_threshold_ms" in data:
        gap = data["sensor_gap_threshold_ms"]
        if not isinstance(gap, (int, float)) or gap <= 0:
            return False, f"sensor_gap_threshold_ms must be positive, got {gap}"

    if "hesitation_threshold_s" in data:
        hes = data["hesitation_threshold_s"]
        if not isinstance(hes, (int, float)) or hes <= 0:
            return False, f"hesitation_threshold_s must be positive, got {hes}"

    if "alignment_angle_threshold_deg" in data:
        align = data["alignment_angle_threshold_deg"]
        if not isinstance(align, (int, float)) or align <= 0 or align > 180:
            return False, f"alignment_angle_threshold_deg must be in (0.0, 180.0], got {align}"

    if "joint_limits" in data and data["joint_limits"] is not None:
        jl = data["joint_limits"]
        if not isinstance(jl, dict):
            return False, "joint_limits must be a dictionary of joint_name: [min, max]"
        for k, v in jl.items():
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                return False, f"joint_limits for '{k}' must be a 2-element tuple/list [min, max]"
            if v[0] >= v[1]:
                return False, f"joint_limits for '{k}' min ({v[0]}) must be strictly less than max ({v[1]})"

    return True, None


def load_config(
    source: Optional[Union[str, Path, Dict[str, Any]]] = None,
    robot_type: Optional[str] = None,
) -> AuditConfig:
    """Load and validate an AuditConfig from file, dictionary, or preset robot profile.
    
    Requirements 12.1, 12.2, 12.3, 12.5.
    
    Args:
        source: Optional file path (JSON) or dictionary of settings.
        robot_type: Optional robot profile preset ('umi', 'franka', 'kinova', 'generic').
        
    Returns:
        Validated AuditConfig instance.
        
    Raises:
        ValueError: If configuration values fail validation.
        FileNotFoundError: If configuration file does not exist.
    """
    base_data: Dict[str, Any] = {}

    # 1. Base from robot profile if specified
    if robot_type:
        rt_lower = robot_type.lower()
        if rt_lower in ROBOT_PROFILES:
            base_data.update(ROBOT_PROFILES[rt_lower])
        else:
            base_data["robot_type"] = robot_type

    # 2. Overlay source data
    if source is not None:
        if isinstance(source, dict):
            data = source
        else:
            p = Path(source)
            if not p.is_file():
                raise FileNotFoundError(f"Configuration file not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Validate overlaid data
        is_valid, error = validate_config_dict(data)
        if not is_valid:
            raise ValueError(f"Invalid configuration settings: {error}")

        # If data specifies robot_type and base wasn't already set, apply profile
        if "robot_type" in data and not base_data:
            prof_name = str(data["robot_type"]).lower()
            if prof_name in ROBOT_PROFILES:
                base_data.update(ROBOT_PROFILES[prof_name])

        base_data.update(data)
    elif not base_data:
        # Default configuration
        return AuditConfig()

    # Re-validate combined dictionary
    is_valid, error = validate_config_dict(base_data)
    if not is_valid:
        raise ValueError(f"Invalid configuration settings: {error}")

    return AuditConfig(
        jitter_threshold=float(base_data.get("jitter_threshold", 0.02)),
        sensor_desync_tolerance_ms=float(base_data.get("sensor_desync_tolerance_ms", 50.0)),
        camera_sync_tolerance_ms=float(base_data.get("camera_sync_tolerance_ms", 16.0)),
        sensor_gap_threshold_ms=float(base_data.get("sensor_gap_threshold_ms", 100.0)),
        hesitation_threshold_s=float(base_data.get("hesitation_threshold_s", 2.0)),
        alignment_angle_threshold_deg=float(base_data.get("alignment_angle_threshold_deg", 15.0)),
        robot_type=str(base_data.get("robot_type", "generic")),
        joint_limits=base_data.get("joint_limits"),
    )


def config_to_dict(config: AuditConfig) -> Dict[str, Any]:
    """Serialize AuditConfig to a dictionary for report metadata.
    
    Requirement 12.4: Include active configuration in audit report metadata section.
    
    Args:
        config: AuditConfig instance.
        
    Returns:
        JSON-serializable dictionary.
    """
    return {
        "jitter_threshold": config.jitter_threshold,
        "sensor_desync_tolerance_ms": config.sensor_desync_tolerance_ms,
        "camera_sync_tolerance_ms": config.camera_sync_tolerance_ms,
        "sensor_gap_threshold_ms": config.sensor_gap_threshold_ms,
        "hesitation_threshold_s": config.hesitation_threshold_s,
        "alignment_angle_threshold_deg": config.alignment_angle_threshold_deg,
        "robot_type": config.robot_type,
        "joint_limits": config.joint_limits,
    }
