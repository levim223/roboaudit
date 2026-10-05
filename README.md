# RoboAudit

Deterministic data quality engine for robotics learning demonstrations.

## Overview

RoboAudit is an open-source, format-agnostic data quality engine for robotics learning demonstrations. The system audits teleoperated demonstrations (including arms, UMI grippers, and multi-camera recordings) for operator mistakes, hardware faults, and logical contradictions. The engine ensures that robotics training datasets meet quality standards before they are used for imitation learning or behavioral cloning.

## Features

- **Deterministic Analysis**: All checks produce identical results given identical inputs (no ML models)
- **Format Agnostic**: Plugin architecture supports arbitrary recording formats
- **Multi-Modal**: Analyzes video streams and telemetry in sync
- **Formal Invariants**: All checks are formally specified for property-based testing
- **Interactive Visualization**: Timeline board for visual exploration of audit results

## Requirements

- Python 3.11 or higher
- Video processing: PyAV (`av`) and OpenCV
- Data analysis: NumPy and Pandas
- Testing: Hypothesis for property-based tests

## Installation

```bash
pip install -e .
```

For development with optional dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from roboaudit import AuditConfig
from roboaudit.core import EpisodeMetadata, ExtractedEpisode

# Configure audit thresholds
config = AuditConfig(
    jitter_threshold=0.02,  # 2.0%
    sensor_desync_tolerance_ms=50.0,
    camera_sync_tolerance_ms=16.0,
    hesitation_threshold_s=2.0
)

# Process episode (implementation pending)
# See tasks.md for development roadmap
```

## Project Structure

```
roboaudit/
├── core/           # Core data models and configuration
├── parsers/        # Episode parsing and format detection
├── analysis/       # Anomaly detection and invariant checking
├── reporting/      # Audit report generation
├── visualization/  # Interactive timeline board
└── tests/          # Test suite
```

## Development Status

This project is under active development. See `.kiro/specs/roboaudit-engine/tasks.md` for the implementation plan.

## License

MIT License
