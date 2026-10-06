---
name: demonstration-audit
description: End-to-end robotics demonstration episode quality audit, hardware timebase verification, formal invariant proof, and interactive timeline board generation.
---

# Robotics Demonstration Quality Audit Skill

This skill guides the AI agent through executing an end-to-end quality audit on robot teleoperation demonstration episodes for imitation learning datasets.

---

## Prerequisites

Before executing this skill:
1. Ensure the `roboaudit` engine is installed and reachable via the MCP server (`@roboaudit`) or Python CLI.
2. Verify that demonstration episodes contain synchronized video streams (e.g., front, wrist) and telemetry time-series (joint positions, velocities, gripper states).

---

## Step 1: Ingestion & Format Verification

1. Discover episode data structure:
   - Identify video files (`front.mp4`, `wrist.mp4`) and telemetry files (`telemetry.csv`, `telemetry.json`, `telemetry.parquet`).
   - Read episode metadata (`metadata.json`) containing task description, instruction, and claimed outcome.
2. Ingest episode using `@roboaudit/audit_episode` or `EpisodeReader`.

---

## Step 2: Hardware Timebase & Multi-Camera Verification

1. Verify PTS timebase regularity:
   - Check video frame intervals against nominal framerate. Jitter $\ge 2.0\%$ triggers `timebase_jitter`.
   - Ensure telemetry timestamps are strictly monotonically increasing.
2. Verify multi-camera synchronization:
   - Ensure timestamps between front and wrist cameras align within $16\,\text{ms}$.
   - Verify feature consistency to detect swapped camera feeds.
3. Check for sensor dropouts:
   - Gaps $> 100\,\text{ms}$ in telemetry stream trigger `sensor_dropout`.
   - Zero sensor variance during commanded motion $> 1.0\,\text{s}$ triggers `sensor_freeze`.

---

## Step 3: Action Phase Segmentation & Grasp Analysis

1. Segment demonstration into chronological phases: `approach`, `grasp`, `manipulate`, `release`, `idle`.
2. Classify contribution of each temporal window: `advancing`, `wasteful`, or `idle`.
3. Inspect gripper state transitions:
   - Detect phantom grasps: gripper closes but optical flow shows zero vertical elevation.
   - Detect missed drops: gripper opens but object fails to transition to receptacle.

---

## Step 4: Operator Mistake Classification & Invariant Proof

1. Identify operator mistakes:
   - Trajectory fumbles: $> 3$ velocity sign reversals per second.
   - Operator hesitation: pauses $> 2.0\,\text{s}$ during non-idle phases.
   - Drops and alignment struggles.
2. Formally prove mathematical invariants:
   - Verify success outcome consistency with goal alignment.
   - Ensure progress monotonicity across advancing phases.
   - Confirm all timestamps stay within $[0, \text{episode\_length\_s}]$.

---

## Step 5: Report & Timeline Board Generation

1. Validate audit report against JSON Schema v1.0.0 using `@roboaudit/validate_report`.
2. Generate interactive HTML5/Canvas Timeline Board using `@roboaudit/visualize_timeline`.
3. Export Markdown summary report.
