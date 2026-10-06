"""Tests for configuration loading, validation, and robot profiles.

Covers:
- Property 48: Configuration Application (Req 12.1)
- Property 49: Invalid Configuration Error Reporting (Req 12.3)
- Unit tests for default thresholds (Req 12.2), metadata export (Req 12.4),
  and robot-specific profiles (Req 12.5)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from roboaudit.core.config import (
    ROBOT_PROFILES,
    config_to_dict,
    load_config,
    validate_config_dict,
)
from roboaudit.core.models import AuditConfig


@st.composite
def valid_config_dict_strategy(draw):
    """Generate valid configuration dictionaries with custom thresholds."""
    jitter = draw(st.floats(min_value=0.001, max_value=0.50))
    desync = draw(st.floats(min_value=1.0, max_value=500.0))
    csync = draw(st.floats(min_value=1.0, max_value=200.0))
    gap = draw(st.floats(min_value=10.0, max_value=1000.0))
    hesitation = draw(st.floats(min_value=0.5, max_value=10.0))
    alignment = draw(st.floats(min_value=1.0, max_value=90.0))
    robot = draw(st.sampled_from(["generic", "umi", "franka", "kinova", "custom_bot"]))

    return {
        "jitter_threshold": jitter,
        "sensor_desync_tolerance_ms": desync,
        "camera_sync_tolerance_ms": csync,
        "sensor_gap_threshold_ms": gap,
        "hesitation_threshold_s": hesitation,
        "alignment_angle_threshold_deg": alignment,
        "robot_type": robot,
    }


@st.composite
def invalid_config_dict_strategy(draw):
    """Generate configuration dictionaries with invalid values."""
    field_to_corrupt = draw(
        st.sampled_from(
            [
                "jitter_threshold",
                "sensor_desync_tolerance_ms",
                "camera_sync_tolerance_ms",
                "hesitation_threshold_s",
                "alignment_angle_threshold_deg",
            ]
        )
    )
    # Generate negative or out-of-range values
    bad_val = draw(st.floats(min_value=-100.0, max_value=-0.001))

    return {field_to_corrupt: bad_val}, field_to_corrupt


class TestConfigProperties:
    """Property-based tests for configuration loading and validation."""

    @settings(max_examples=100)
    @given(cfg_dict=valid_config_dict_strategy())
    def test_property_48_configuration_application(self, cfg_dict: dict):
        """Property 48: Configuration Application.
        
        Validates Requirements: 12.1.
        The Audit_Engine SHALL accept a configuration specifying jitter thresholds,
        timeout values, and custom settings, and apply specified values accurately.
        """
        config = load_config(cfg_dict)

        assert isinstance(config, AuditConfig)
        assert abs(config.jitter_threshold - cfg_dict["jitter_threshold"]) < 1e-5
        assert abs(config.sensor_desync_tolerance_ms - cfg_dict["sensor_desync_tolerance_ms"]) < 1e-5
        assert abs(config.camera_sync_tolerance_ms - cfg_dict["camera_sync_tolerance_ms"]) < 1e-5
        assert abs(config.sensor_gap_threshold_ms - cfg_dict["sensor_gap_threshold_ms"]) < 1e-5
        assert abs(config.hesitation_threshold_s - cfg_dict["hesitation_threshold_s"]) < 1e-5
        assert abs(config.alignment_angle_threshold_deg - cfg_dict["alignment_angle_threshold_deg"]) < 1e-5
        assert config.robot_type == cfg_dict["robot_type"]

    @settings(max_examples=100)
    @given(data=invalid_config_dict_strategy())
    def test_property_49_invalid_configuration_error_reporting(self, data):
        """Property 49: Invalid Configuration Error Reporting.
        
        Validates Requirements: 12.3.
        The Audit_Engine SHALL validate configuration files and report descriptive errors
        for invalid settings.
        """
        bad_dict, corrupted_field = data
        is_valid, error = validate_config_dict(bad_dict)

        assert is_valid is False
        assert error is not None
        assert corrupted_field in error

        with pytest.raises(ValueError, match=corrupted_field):
            load_config(bad_dict)


class TestConfigUnit:
    """Unit tests for configuration defaults, file loading, and robot profiles."""

    def test_default_configuration_values(self):
        """Requirement 12.2: Use documented default threshold values when none provided."""
        config = load_config()
        assert config.jitter_threshold == 0.02  # 2.0%
        assert config.sensor_desync_tolerance_ms == 50.0  # 50ms
        assert config.camera_sync_tolerance_ms == 16.0  # 16ms
        assert config.sensor_gap_threshold_ms == 100.0  # 100ms
        assert config.hesitation_threshold_s == 2.0  # 2.0s
        assert config.alignment_angle_threshold_deg == 15.0  # 15 deg
        assert config.robot_type == "generic"

    def test_robot_profiles(self):
        """Requirement 12.5: Support per-robot-type configuration profiles."""
        for prof_name in ["umi", "franka", "kinova"]:
            config = load_config(robot_type=prof_name)
            assert config.robot_type == prof_name
            if prof_name in ["franka", "kinova"]:
                assert config.joint_limits is not None
                assert "joint_1" in config.joint_limits

        umi_cfg = load_config(robot_type="umi")
        assert umi_cfg.hesitation_threshold_s == 1.5

    def test_config_to_dict_metadata_export(self):
        """Requirement 12.4: Include active configuration in audit report metadata."""
        config = load_config(robot_type="franka")
        meta = config_to_dict(config)

        assert meta["robot_type"] == "franka"
        assert meta["jitter_threshold"] == 0.02
        assert "joint_limits" in meta
        # Ensure json-serializable
        json_str = json.dumps(meta)
        assert len(json_str) > 0

    def test_load_from_json_file(self, tmp_path: Path):
        cfg_file = tmp_path / "custom_config.json"
        data = {
            "jitter_threshold": 0.03,
            "sensor_desync_tolerance_ms": 75.0,
            "robot_type": "kinova",
        }
        cfg_file.write_text(json.dumps(data))

        config = load_config(cfg_file)
        assert config.jitter_threshold == 0.03
        assert config.sensor_desync_tolerance_ms == 75.0
        assert config.robot_type == "kinova"
