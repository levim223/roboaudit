---
inclusion: always
---

# Coding Standards for RoboAudit Engine

This document defines the Python coding standards, testing practices, and architectural constraints for the RoboAudit engine implementation.

## Python Language Standards

### Version and Compatibility

- **Python Version**: 3.11+ (required for modern type hints and performance improvements)
- **Future Imports**: Always include rom __future__ import annotations at the top of every module for forward-compatible type hints
- **No Backward Compatibility**: Code does not need to support Python < 3.11

### Type Hinting (Strict)

All functions, methods, and public APIs MUST have complete type annotations:

`python
from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Literal, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ✅ GOOD: Full type hints
def calculate_jitter(
    timestamps: np.ndarray,
    expected_interval: float,
    threshold: float = 0.02
) -> List[TimebaseIssue]:
    """
    Detect frame jitter in timestamp sequence.
    
    Args:
        timestamps: Array of frame timestamps in seconds
        expected_interval: Expected time between frames (1/fps)
        threshold: Jitter threshold as fraction (default 0.02 = 2%)
    
    Returns:
        List of detected timebase issues
    """
    issues: List[TimebaseIssue] = []
    # Implementation...
    return issues

# ❌ BAD: Missing type hints
def calculate_jitter(timestamps, expected_interval, threshold=0.02):
    issues = []
    return issues
`

**Required Type Annotations**:
- Function parameters (including self in return types if returning instance)
- Function return values (use None explicitly, not omit)
- Class attributes (in __init__ or as dataclass fields)
- Module-level constants

**Type Hint Guidelines**:
- Use Optional[T] for nullable values (equivalent to T | None in Python 3.10+)
- Use Literal["value1", "value2"] for string enums and restricted values
- Use List[T], Dict[K, V], Tuple[T, ...] for collections (not list, dict, 	uple)
- Use 
p.ndarray for NumPy arrays; add shape/dtype in docstring if critical
- Use pd.DataFrame for pandas DataFrames; document expected columns in docstring

### Data Modeling with Dataclasses

All data structures MUST use @dataclass (or pydantic.BaseModel for validation-heavy cases):

`python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class TimebaseIssue:
    """Represents a temporal inconsistency in episode data."""
    issue_type: Literal["frame_jitter", "telemetry_jitter", "non_monotonic", "sensor_desync"]
    severity: Literal["warning", "error"]
    time_range: Tuple[float, float]  # (start_s, end_s)
    affected_stream: str
    details: str

@dataclass
class TemporalWindow:
    """Represents a classified segment of the episode timeline."""
    start_s: float
    end_s: float
    action_phase: Literal["approach", "grasp", "manipulate", "release", "idle"]
    arm_attribution: Literal["left", "right", "both", "none"]
    contribution_type: Literal["advancing", "wasteful", "idle"]
    completion_percentage: float  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate field constraints after initialization."""
        if not (0.0 <= self.completion_percentage <= 1.0):
            raise ValueError(f"completion_percentage must be in [0.0, 1.0], got {self.completion_percentage}")
        if self.start_s >= self.end_s:
            raise ValueError(f"start_s ({self.start_s}) must be < end_s ({self.end_s})")

@dataclass
class AuditConfig:
    """Configuration for audit thresholds and robot-specific parameters."""
    jitter_threshold: float = 0.02  # 2.0%
    sensor_desync_tolerance_ms: float = 50.0
    camera_sync_tolerance_ms: float = 16.0
    sensor_gap_threshold_ms: float = 100.0
    hesitation_threshold_s: float = 2.0
    alignment_angle_threshold_deg: float = 15.0
    robot_type: str = "generic"
    joint_limits: Optional[Dict[str, Tuple[float, float]]] = None
`

**Dataclass Guidelines**:
- ✅ Use @dataclass for data-only classes (no complex methods)
- ✅ Use rozen=True for immutable data structures where appropriate
- ✅ Implement __post_init__ for field validation if needed
- ✅ Document all fields with inline comments or docstrings
- ❌ Avoid mutable default values (use ield(default_factory=list) instead of = [])

### Functional Purity for Invariant Checks

All invariant verification and deterministic check functions MUST be pure functions with zero I/O side effects:

**Pure Function Rules**:
1. **Deterministic**: Same inputs always produce same outputs
2. **No Side Effects**: No file I/O, no network calls, no global state mutation, no logging inside pure functions
3. **No External Dependencies**: Only depend on input parameters (no database queries, no API calls)
4. **Explicit Return**: Always return structured violation objects or None

`python
# ✅ GOOD: Pure invariant check
def check_progress_vs_outcome(report: AuditReport) -> Optional[InvariantViolation]:
    """
    Verify that failure/partial outcomes have progress < 1.0.
    Pure function: no side effects, deterministic output.
    """
    outcome = report.completion.outcome
    max_progress = max((w.progress for w in report.timeline.windows), default=0.0)
    
    if outcome in ["failure", "partial"] and max_progress >= 1.0:
        return InvariantViolation(
            invariant_name="progress_vs_outcome",
            severity="error",
            details=f"Task outcome is '{outcome}' but timeline progress reaches {max_progress:.2f}",
            affected_data={"outcome": outcome, "max_progress": max_progress}
        )
    return None

# ❌ BAD: Impure function with I/O side effects
def check_progress_vs_outcome(report: AuditReport) -> Optional[InvariantViolation]:
    outcome = report.completion.outcome
    max_progress = max((w.progress for w in report.timeline.windows), default=0.0)
    
    if outcome in ["failure", "partial"] and max_progress >= 1.0:
        # ❌ Side effect: logging
        logger.error(f"Invariant violation: {outcome} with progress {max_progress}")
        
        # ❌ Side effect: file I/O
        with open("violations.log", "a") as f:
            f.write(f"Violation at {datetime.now()}\n")
        
        return InvariantViolation(...)
    return None
`

**Where to Handle Side Effects**:
- Logging: In the caller (e.g., mit_audit_report() logs violations after collecting them)
- File I/O: In orchestration layer, never in invariant checks
- Error handling: Return violation objects; caller decides whether to raise exceptions

### Error Handling

Use exceptions for unrecoverable errors; use Result types or Optional for expected failure modes:

`python
from typing import Union, Optional
from dataclasses import dataclass

# For expected failures: use Result pattern
@dataclass
class Ok:
    value: Any

@dataclass
class Err:
    error: str

Result = Union[Ok, Err]

def parse_telemetry(file_path: Path) -> Result[pd.DataFrame, str]:
    """
    Parse telemetry file into DataFrame.
    Returns Result for expected failures (file not found, parse errors).
    """
    try:
        if not file_path.exists():
            return Err(f"Telemetry file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if "timestamp_us" not in df.columns:
            return Err("Missing required column: timestamp_us")
        
        return Ok(df)
    
    except Exception as e:
        return Err(f"Failed to parse telemetry: {e}")

# For unexpected errors: raise exceptions
def calculate_jitter(timestamps: np.ndarray, threshold: float) -> List[TimebaseIssue]:
    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamps to calculate jitter")
    
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError(f"Threshold must be in (0, 1), got {threshold}")
    
    # Implementation...
`

### Code Organization

`
roboaudit/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── models.py          # Dataclasses: AuditReport, Timeline, TemporalWindow, etc.
│   ├── config.py          # AuditConfig, configuration loading
│   └── types.py           # Type aliases, enums, constants
├── parsers/
│   ├── __init__.py
│   ├── plugin.py          # ParserPlugin ABC, ParserPluginRegistry
│   ├── episode_reader.py  # EpisodeReader
│   ├── video_extractor.py # PyAV frame extraction
│   └── telemetry_extractor.py
├── analysis/
│   ├── __init__.py
│   ├── timebase.py        # TimebaseVerifier
│   ├── camera_sync.py     # CameraSyncChecker
│   ├── grasp_anomaly.py   # GraspAnomalyDetector
│   ├── phase_segmenter.py # ActionPhaseSegmenter
│   ├── sensor_monitor.py  # SensorDropoutMonitor
│   └── operator_mistakes.py
├── invariants/
│   ├── __init__.py
│   ├── checker.py         # InvariantChecker, verify_all_invariants()
│   └── violations.py      # InvariantViolation dataclass
├── reporting/
│   ├── __init__.py
│   ├── generator.py       # AuditReportGenerator
│   ├── parser.py          # AuditParser, schema validation
│   └── exporter.py        # JSON and Markdown exporters
├── visualization/
│   ├── __init__.py
│   ├── timeline_board.py  # HTML5/Canvas timeline generator
│   └── server.py          # Lightweight HTTP server
└── cli/
    ├── __init__.py
    └── main.py            # Command-line interface
`

---

## Testing Standards

### Property-Based Testing with Hypothesis

Every invariant MUST have a property-based test running ≥100 iterations:

`python
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import integers, floats, sampled_from, lists, builds

# Custom strategies for domain objects
@st.composite
def temporal_windows(
    draw,
    contribution: Optional[str] = None,
    progress_pattern: Optional[str] = None
):
    """Generate TemporalWindow instances with controlled properties."""
    start_s = draw(floats(min_value=0.0, max_value=100.0))
    end_s = draw(floats(min_value=start_s + 0.1, max_value=start_s + 10.0))
    action = draw(sampled_from(["approach", "grasp", "manipulate", "release", "idle"]))
    arm = draw(sampled_from(["left", "right", "both", "none"]))
    
    if contribution:
        contrib = contribution
    else:
        contrib = draw(sampled_from(["advancing", "wasteful", "idle"]))
    
    if progress_pattern == "decreasing":
        # Generate decreasing progress for monotonicity violation tests
        progress = draw(floats(min_value=0.0, max_value=1.0))
    else:
        progress = draw(floats(min_value=0.0, max_value=1.0))
    
    return TemporalWindow(
        start_s=start_s,
        end_s=end_s,
        action_phase=action,
        arm_attribution=arm,
        contribution_type=contrib,
        completion_percentage=progress
    )

@st.composite
def timelines_with_decreasing_progress(draw):
    """Generate Timeline with progress violations for testing."""
    windows = []
    progress = 1.0  # Start high
    
    for _ in range(draw(integers(min_value=2, max_value=5))):
        window = draw(temporal_windows(contribution="advancing"))
        window.completion_percentage = progress
        windows.append(window)
        progress -= 0.2  # Decrease (violates monotonicity)
    
    return Timeline(windows=windows)

# Property-based test
@given(timelines_with_decreasing_progress())
@settings(max_examples=100, deadline=None)
def test_progress_monotonicity_property(timeline: Timeline):
    """
    Property: Progress must not decrease during advancing phases.
    This test generates timelines with decreasing progress to verify detection.
    """
    violation = check_progress_monotonicity(timeline)
    assert violation is not None, "Expected invariant violation for decreasing progress"
    assert violation.invariant_name == "progress_monotonicity"
    assert violation.severity == "error"

@given(
    outcome=sampled_from(["failure", "partial"]),
    max_progress=floats(min_value=1.0, max_value=1.0)  # Always 1.0
)
@settings(max_examples=100)
def test_progress_vs_outcome_property(outcome: str, max_progress: float):
    """
    Property: failure/partial outcomes cannot have progress >= 1.0.
    """
    # Build minimal report for testing
    report = AuditReport(
        schema_version="1.0.0",
        context=EpisodeContext(...),
        timeline=Timeline(windows=[TemporalWindow(..., completion_percentage=max_progress)]),
        completion=TaskCompletion(outcome=outcome, ...),
        ...
    )
    
    violation = check_progress_vs_outcome(report)
    assert violation is not None
    assert violation.invariant_name == "progress_vs_outcome"
`

**Hypothesis Configuration**:
`python
# In conftest.py or test module
from hypothesis import settings

# Default settings for all property tests
settings.register_profile("roboaudit", max_examples=100, deadline=5000)
settings.load_profile("roboaudit")
`

### Unit Testing Guidelines

`python
import pytest
from pathlib import Path

# ✅ GOOD: Descriptive test names, arrange-act-assert pattern
def test_frame_jitter_detection_flags_warning_when_variance_exceeds_threshold():
    # Arrange
    fps = 30.0
    expected_interval = 1.0 / fps
    # Create timestamps with 3% jitter (exceeds 2% threshold)
    timestamps = np.array([0.0, 0.0333, 0.0666, 0.1000, 0.1350])  # Last interval has 3% error
    verifier = TimebaseVerifier()
    
    # Act
    issues = verifier.check_video_timebase(timestamps, fps, jitter_threshold=0.02)
    
    # Assert
    assert len(issues) == 1
    assert issues[0].issue_type == "frame_jitter"
    assert issues[0].severity == "warning"

# ✅ GOOD: Parametrized tests for multiple cases
@pytest.mark.parametrize("outcome,max_progress,expect_violation", [
    ("failure", 1.0, True),
    ("partial", 0.99, False),
    ("success", 1.0, False),
    ("failure", 0.95, False),
])
def test_progress_vs_outcome_invariant(outcome, max_progress, expect_violation):
    report = create_test_report(outcome=outcome, max_progress=max_progress)
    violation = check_progress_vs_outcome(report)
    
    if expect_violation:
        assert violation is not None
    else:
        assert violation is None
`

### Integration Testing

`python
import tempfile
from pathlib import Path

def test_end_to_end_audit_workflow():
    """Integration test: complete audit from episode directory to report."""
    # Arrange: create test episode with fixture data
    with tempfile.TemporaryDirectory() as tmpdir:
        episode_dir = Path(tmpdir) / "test_episode"
        create_test_episode(episode_dir)  # Helper creates videos + telemetry
        
        config = AuditConfig(robot_type="umi")
        
        # Act: run full audit
        reader = EpisodeReader()
        episode = reader.read_episode(episode_dir, config)
        
        analyzer = AuditEngine(config)
        report = analyzer.audit_episode(episode)
        
        # Assert: verify report structure
        assert report.schema_version == "1.0.0"
        assert len(report.timeline.windows) > 0
        assert 0.0 <= report.quality_metrics.quality_score <= 1.0
        
        # Verify invariants
        violations = verify_all_invariants(report)
        assert len(violations) == 0, f"Report has invariant violations: {violations}"
`

### Test Organization

`
tests/
├── unit/
│   ├── test_timebase.py
│   ├── test_invariants.py
│   ├── test_grasp_anomaly.py
│   └── ...
├── property/
│   ├── test_invariant_properties.py
│   ├── test_serialization_properties.py
│   └── strategies.py  # Hypothesis strategies
├── integration/
│   ├── test_episode_ingestion.py
│   ├── test_full_audit_workflow.py
│   └── ...
├── fixtures/
│   ├── sample_episodes/
│   ├── telemetry_samples/
│   └── ...
└── conftest.py  # Pytest fixtures and configuration
`

---

## Code Quality Standards

### Linting and Formatting

- **Formatter**: lack with 100-character line length
- **Linter**: uff (replaces flake8, isort, pylint)
- **Type Checker**: mypy in strict mode

**Configuration** (pyproject.toml):
`	oml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "UP", "ANN", "B", "A", "C4", "RET", "SIM"]
ignore = ["ANN101", "ANN102"]  # Ignore self/cls annotations

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
`

### Documentation

- **Docstrings**: Google style for all public functions, classes, and modules
- **Type Hints**: Serve as inline documentation; docstrings add context
- **README**: Installation, usage examples, configuration reference

`python
def check_progress_monotonicity(timeline: Timeline) -> Optional[InvariantViolation]:
    """
    Verify that completion progress is monotonically non-decreasing during advancing phases.
    
    This invariant ensures logical consistency in timeline construction: if the robot
    is actively advancing toward the task goal, progress cannot decrease. Similarly,
    during idle phases, progress must remain constant.
    
    Args:
        timeline: Timeline containing sequence of temporal windows with progress values.
    
    Returns:
        InvariantViolation if progress decreases during advancing or increases during idle.
        None if timeline satisfies monotonicity constraints.
    
    Raises:
        ValueError: If timeline has fewer than 1 window (cannot check monotonicity).
    
    Example:
        >>> windows = [
        ...     TemporalWindow(..., contribution_type="advancing", completion_percentage=0.5),
        ...     TemporalWindow(..., contribution_type="advancing", completion_percentage=0.8),
        ... ]
        >>> timeline = Timeline(windows=windows)
        >>> violation = check_progress_monotonicity(timeline)
        >>> assert violation is None  # Valid monotonic increase
    """
    if len(timeline.windows) < 1:
        raise ValueError("Timeline must have at least 1 window")
    
    # Implementation...
`

---

## Architectural Constraints

### Zero External API Dependencies

- ❌ No calls to external APIs during invariant checks or core analysis
- ❌ No ML model inference (no TensorFlow, PyTorch, etc.)
- ✅ All checks must be deterministic and runnable offline

### Determinism Requirement

- All analysis functions must produce identical outputs given identical inputs
- No randomness, no timestamps, no UUIDs in deterministic paths
- Use fixed seeds for any necessary randomness in tests

### Performance Considerations

- Use NumPy vectorized operations over Python loops for numerical computations
- Use pandas for time-series analysis with efficient indexing
- Avoid premature optimization; profile before optimizing

---

## Summary

**Key Standards**:
1. ✅ Python 3.11+ with strict type hints (rom __future__ import annotations)
2. ✅ Dataclasses for all data structures with validation
3. ✅ Pure functions for invariant checks (no side effects)
4. ✅ Property-based testing with Hypothesis (≥100 iterations per property)
5. ✅ Zero external API dependencies
6. ✅ Deterministic, reproducible analysis
7. ✅ Google-style docstrings for public APIs
8. ✅ Black formatting, Ruff linting, Mypy type checking

These standards ensure RoboAudit is testable, maintainable, and produces reliable, reproducible results.
