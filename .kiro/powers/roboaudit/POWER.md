---
name: roboaudit
displayName: RoboAudit Quality Engine
description: Deterministic robotics demonstration episode quality, hardware timebase, formal invariant, and operator mistake auditing engine
keywords:
  - robotics
  - imitation learning
  - quality audit
  - timebase jitter
  - sensor dropout
  - camera synchronization
  - grasp anomaly
  - timeline board
  - mcp
---

# RoboAudit Quality Engine Power

The **RoboAudit Power** provides Kiro agents with specialized domain intelligence, mathematical invariant verification, hardware synchronization checks, and automated audit tools for robot demonstration datasets.

---

## 1. Purpose & Scope

This power equips agents to:
1. **Audit Episode Timebases**: Detect camera frame PTS jitter (threshold $\ge 2.0\%$), telemetry sampling gaps ($> 100\,\text{ms}$), sensor freezes ($> 1.0\,\text{s}$), and cross-modal video-telemetry desynchronization ($> 50\,\text{ms}$).
2. **Verify Camera Synchronicity**: Confirm multi-camera phase offsets remain within $16\,\text{ms}$ and detect inadvertent camera channel swaps.
3. **Detect Grasp Anomalies**: Identify phantom grasps and missed drops via optical flow elevation delta measurement.
4. **Classify Operator Mistakes**: Quantify operator hesitation ($> 2.0\,\text{s}$), fumbles ($> 3$ velocity reversals/s), and alignment struggles.
5. **Verify Formal Invariants**: Formally prove task progress monotonicity, time bounds, and outcome consistency against goal alignments.
6. **Generate Interactive Timeline Boards**: Produce self-contained HTML5/Canvas visualization boards with interactive scrubbing, playback, and anomaly inspection.

---

## 2. Activation Signals

This power activates automatically when the user or prompt mentions:
- Robotic demonstration datasets, teleoperation traces, or imitation learning data.
- Video stream PTS jitter, dropped frames, or camera desync.
- Telemetry dropouts, sensor freezes, or gripper state anomalies.
- Formal invariant verification on robot trajectories.
- Episode quality scores, batch auditing, or Timeline Board generation.

---

## 3. Strict Invariant Guardrails

When this power is active, agents **MUST** enforce the following 5 fundamental mathematical invariants:
1. **Invariant 1 (`outcome_vs_alignment`)**: Success outcomes require `goal_alignment.relation = "aligned"`.
2. **Invariant 2 (`progress_vs_outcome`)**: Failure and partial outcomes cannot reach $1.0$ completion.
3. **Invariant 3 (`undone_timing`)**: In `success_then_undone` outcomes, `undone_at_s > goal_reached_at_s`.
4. **Invariant 4 (`progress_monotonicity`)**: Progress is non-decreasing during advancing phases and constant during idle phases.
5. **Invariant 5 (`time_bounds`)**: All event timestamps must reside strictly within $[0.0, \text{episode\_length\_s}]$.
