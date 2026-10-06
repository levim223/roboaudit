# RoboAudit

> **Deterministic data quality & invariant auditing engine for robotics demonstration datasets.**

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-124%20passed-success.svg)](https://github.com/houssamm26/roboaudit)
[![Properties](https://img.shields.io/badge/hypothesis-27%20properties%20%7C%20%E2%89%A5100%20iter-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![MCP](https://img.shields.io/badge/MCP-2024--11--05-blueviolet.svg)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

RoboAudit is an industrial, format-agnostic quality engine engineered to audit robotic teleoperation and imitation learning demonstrations (including bimanual arms, UMI grippers, and multi-camera rigs). It identifies silent hardware desynchronization, video PTS jitter, sensor dropouts, grasp anomalies, operator fumbles, and mathematical invariant contradictions before corrupted data enters policy training pipelines.

---

## Key Capabilities

* **Deterministic Analysis**: Zero ML models in the critical path; 100% reproducible, mathematically provable invariant checks.
* **Hardware Timebase Verification**: Video presentation timestamp (PTS) regularity checks (jitter $\ge 2.0\%$), telemetry monotonic ordering, and sensor dropout detection ($> 100\,\text{ms}$).
* **Multi-Camera Synchronization**: Sub-16ms multi-view phase alignment and automated camera feed inversion/swap detection.
* **Action Phase Segmentation**: Autonomous segmentation into `approach`, `grasp`, `manipulate`, `release`, and `idle` windows with contribution classification (`advancing`, `wasteful`, `idle`).
* **Physical & Operator Anomaly Detection**: Optical flow vertical elevation measurement for phantom grasps, trajectory velocity reversal counting ($> 3\,\text{rev/s}$) for alignment fumbles, and operator hesitation pauses ($> 2.0\,\text{s}$).
* **Interactive Timeline Board**: Standalone HTML5/Canvas visualization board featuring playhead scrubbing, multi-track timeline alignment, color-coded contribution phases, and click-to-inspect anomaly markers.
* **High-Throughput Batch Processing**: Multi-threaded parallel dataset audits with fault isolation and aggregate quality analytics.

---

## Architecture Overview

```
                                      RoboAudit Engine
                                              │
    ┌─────────────────────────────── Ingestion Pipeline ──────────────────────────────┐
    │                                                                                  │
    │   Multi-Camera MP4 Streams               Telemetry Streams (CSV/JSON/Parquet)    │
    │     [PyAV PTS Decoder]                       [TimeSeries Normalizer]             │
    └───────────────────────┬──────────────────────────────────┬───────────────────────┘
                            │                                  │
    ┌───────────────────────┴──────── Verification Pipeline ───┴───────────────────────┐
    │                                                                                  │
    │   • TimebaseVerifier      (Frame Jitter ≥ 2.0%, Telemetry Drift, Sensor Desync) │
    │   • CameraSyncDetector    (Multi-view Phase Offset ≤ 16ms, Camera Swap Check)    │
    │   • SensorDropoutMonitor  (Dropouts > 100ms, Sensor Freeze > 1.0s, Outliers)     │
    │   • ActionPhaseSegmenter  (Chronological Windows, Contribution Classification)   │
    │   • GraspAnomalyDetector  (Phantom Grasps, Missed Drops, Optical Flow Delta)    │
    │   • OperatorMistakes      (Hesitations > 2.0s, Trajectory Fumbles, Drops)        │
    │   • InvariantChecker      (Outcome Consistency, Monotonicity, Time Bounds)       │
    │   • AlignmentScorer       (Goal Scoring, Penalty Bounds, Failure Outcome Cap)   │
    └─────────────────────────────────────────┬────────────────────────────────────────┘
                                              │
    ┌──────────────────────────────── Reporting & Export ──────────────────────────────┐
    │                                                                                  │
    │   • JSON Report (Schema v1.0.0)         • Human-Readable Markdown Report         │
    │   • Interactive HTML5 Timeline Board    • Batch Summary Analytics               │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/houssamm26/roboaudit.git
cd roboaudit
pip install -e .
```

For development and test dependencies:

```bash
pip install -e ".[dev]"
```

---

## Quick Start (Python API)

```python
from roboaudit import RoboAuditEngine, AuditConfig, load_config

# 1. Initialize engine with preset robot platform profile ('umi', 'franka', 'kinova', or 'generic')
config = load_config(robot_type="umi")
engine = RoboAuditEngine(config=config)

# 2. Audit a single demonstration episode
report = engine.audit_episode("sample_data/demo_episode")

# 3. Inspect results
print(f"Episode ID:     {report.context.episode_id}")
print(f"Quality Score:  {report.quality_metrics.quality_score:.2f} / 1.00")
print(f"Task Completed: {report.completion.task_completed}")
print(f"Data Issues:    {len(report.data_issues)}")
print(f"Mistakes:       {len(report.operator_mistakes)}")

# 4. Generate interactive Timeline Board
from roboaudit import TimelineBoardGenerator
board = TimelineBoardGenerator()
board.generate_html(report, "timeline_board.html")
```

---

## Command-Line Interface (CLI)

RoboAudit includes a unified command-line interface:

### 1. Audit Single Episode
```bash
roboaudit audit sample_data/demo_episode
```
Output:
```
========================================================
 RoboAudit Quality Report: demo_episode
========================================================
 Dataset:           default
 Rig:               generic_robot
 Instruction:       Execute task
 Duration:          11.00s
 Quality Score:     0.93 / 1.00
 Goal Alignment:    1.00 (aligned)
 Task Completed:    True
 Total Anomalies:   2
 Data Issues:       2
 Operator Mistakes: 0
========================================================
```

Options:
```bash
# Save structured JSON and Markdown reports
roboaudit audit sample_data/demo_episode --output report.json --markdown report.md --html board.html
```

### 2. Batch Processing
```bash
roboaudit batch path/to/dataset/ --workers 8 --output batch_summary.json --markdown batch_summary.md
```

### 3. Generate & Serve Interactive Timeline Board
```bash
roboaudit visualize sample_data/demo_episode --output board.html --serve --port 8080
```

### 4. Validate Audit Report
```bash
roboaudit validate sample_data/report.json
```

---

## Formal Invariants Enforced

RoboAudit verifies five mathematical invariants across every demonstration:

1. **Outcome vs. Alignment (`outcome_vs_alignment`)**:
   An episode claiming a `success` outcome must have `goal_alignment.relation == "aligned"`.
2. **Progress vs. Outcome (`progress_vs_outcome`)**:
   Episodes with `failure` or `partial` outcomes cannot reach $\ge 1.0$ completion.
3. **Temporal Ordering of Undone Tasks (`undone_timing`)**:
   In `success_then_undone` demonstrations, `undone_at_s` must strictly follow `goal_reached_at_s`.
4. **Progress Monotonicity (`progress_monotonicity`)**:
   During `advancing` action phases, completion progress is strictly non-decreasing ($\frac{dP}{dt} \ge 0$). During `idle` phases, progress is constant.
5. **Time Boundary Containment (`time_bounds`)**:
   All event and window timestamps must satisfy $0.0 \le t \le \text{episode\_length\_s}$.

---

## Kiro Ecosystem Integration

RoboAudit is built to seamlessly integrate with Kiro tools:

| System | Configuration | Description |
|---|---|---|
| **Spec-Driven Development** | [`.kiro/specs/roboaudit-engine/`](.kiro/specs/roboaudit-engine/) | Formal EARS requirements (`requirements.md`), technical architecture (`design.md`), and task milestone tracking (`tasks.md`). |
| **Steering Documents** | [`.kiro/steering/`](.kiro/steering/) | Persistent domain rules for Python coding standards, robotics data standards, and mathematical invariants. |
| **Pre-Task Hooks** | [`.kiro/hooks/`](.kiro/hooks/) | Automated validation hooks enforcing invariant checking and standards guards prior to task execution. |
| **Property-Based Testing** | [`roboaudit/tests/strategies.py`](roboaudit/tests/strategies.py) | Hypothesis strategy generators running $\ge 100$ iterations per property to prove state and timeline correctness. |
| **Native Power Bundle** | [`.kiro/powers/roboaudit/`](.kiro/powers/roboaudit/) | Packaged Power manifest (`plugin.json`), activation rules (`POWER.md`), and demonstration audit skills (`SKILL.md`). |
| **Model Context Protocol (MCP)** | [`.kiro/settings/mcp.json`](.kiro/settings/mcp.json) | Local stdio server exposing 4 tools: `audit_episode`, `batch_audit`, `visualize_timeline`, `validate_report`. |
| **Custom Agent** | [`.kiro/agents/robotics-auditor.json`](.kiro/agents/robotics-auditor.json) | Dedicated autonomous auditor agent (`Ctrl + Shift + R`) equipped with `@roboaudit` MCP tools. |

---

## Project Structure

```
roboaudit/
├── .kiro/
│   ├── agents/               # Custom agent definition (Robotics Auditor)
│   ├── hooks/                # Pre-task and invariant verification hooks
│   ├── powers/               # Packaged RoboAudit Power bundle
│   ├── settings/             # Workspace MCP server configuration
│   ├── specs/                # EARS specifications, design docs, task tracker
│   └── steering/             # Coding standards, hardware thresholds, invariants
├── roboaudit/
│   ├── analysis/             # Detection: timebase, camera sync, dropout, grasp, mistakes, invariants
│   ├── core/                 # Dataclasses, JSON schema v1.0.0, configuration & robot profiles
│   ├── parsers/              # EpisodeReader, video/telemetry decoders, plugin registry
│   ├── reporting/            # JSON/Markdown report generators and schema validator
│   ├── visualization/        # Interactive HTML5/Canvas Timeline Board generator
│   ├── batch.py              # Parallel batch execution and summary generator
│   ├── cli.py                # Command-line interface (audit, batch, visualize, validate)
│   ├── engine.py             # Unified end-to-end audit orchestrator
│   └── server.py             # FastMCP stdio server
├── scripts/
│   └── mcp_server.py         # Standalone MCP server launcher
├── sample_data/
│   └── demo_episode/         # Demonstration fixtures with telemetry and multi-view frames
└── tests/                    # 124 unit and property-based test suites (Hypothesis)
```

---

## Verification & Testing

Run the full test suite (124 tests including 27 property tests):

```bash
python -m pytest
```

Run property-based tests with randomized seeds:

```bash
python -m pytest roboaudit/tests/test_properties.py --hypothesis-seed=random
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
