---
inclusion: always
---

# Audit Invariants for RoboAudit Engine

This document defines the formal mathematical invariants that govern timeline validation and logical consistency in the RoboAudit engine. Every audit report MUST satisfy these invariants before emission.

## Overview

Invariants are universally quantified logical properties that MUST hold across all valid audit reports. Violations indicate:
- Mislabeled demonstrations (human annotation errors)
- Logical contradictions in episode data
- Incomplete or corrupted timeline construction

All invariants are implemented as deterministic verification functions with zero I/O side effects, returning structured violation objects.

---

## Core Invariants

### Invariant 1: Outcome vs Alignment (outcome_vs_alignment)

**Formal Statement**:
```
For all audit_report: 
  (outcome = "success") implies (goal_alignment.relation = "aligned")
```

**Plain English**: If a task outcome is marked as success, the goal alignment relation MUST be aligned. A demonstration cannot be successful if the operator performed a different or unrelated task.

**Violation Conditions**:
- outcome = "success" AND goal_alignment.relation in {"different", "unrelated", "contradictory"}

**Implementation**:
```python
def check_outcome_vs_alignment(report: AuditReport) -> Optional[InvariantViolation]:
    if report.completion.outcome == "success":
        if report.goal_alignment.relation != "aligned":
            return InvariantViolation(
                invariant_name="outcome_vs_alignment",
                severity="error",
                details=f"Task marked success but goal_alignment.relation is '{report.goal_alignment.relation}' (expected 'aligned')",
                affected_data={
                    "outcome": report.completion.outcome,
                    "goal_alignment": report.goal_alignment.relation
                }
            )
    return None
```

---

### Invariant 2: Progress vs Outcome (progress_vs_outcome)

**Formal Statement**:
```
For all audit_report:
  (outcome in {"failure", "partial"}) implies (max(timeline.progress) < 1.0)
```

**Plain English**: If a task outcome is failure or partial, the timeline completion percentage MUST NOT reach 1.0 (100% complete). Failed or partially completed tasks cannot have full progression.

**Violation Conditions**:
- outcome in {"failure", "partial"} AND max(window.progress for window in timeline) >= 1.0

**Implementation**:
```python
def check_progress_vs_outcome(report: AuditReport) -> Optional[InvariantViolation]:
    outcome = report.completion.outcome
    max_progress = max((w.progress for w in report.timeline.windows), default=0.0)
    
    if outcome in ["failure", "partial"] and max_progress >= 1.0:
        return InvariantViolation(
            invariant_name="progress_vs_outcome",
            severity="error",
            details=f"Task outcome is '{outcome}' but timeline progress reaches {max_progress:.2f} (must be < 1.0)",
            affected_data={
                "outcome": outcome,
                "max_progress": max_progress
            }
        )
    return None
```

---

### Invariant 3: Undone Timing (undone_timing)

**Formal Statement**:
```
For all audit_report:
  (outcome = "success_then_undone") implies (undone_at_s > goal_reached_at_s)
```

**Plain English**: If a task outcome is success_then_undone (goal achieved then reversed), the undo timestamp MUST be strictly greater than the goal achievement timestamp.

**Violation Conditions**:
- outcome = "success_then_undone" AND undone_at_s <= goal_reached_at_s

**Implementation**:
```python
def check_undone_timing(report: AuditReport) -> Optional[InvariantViolation]:
    completion = report.completion
    
    if completion.outcome == "success_then_undone":
        if completion.undone_at_s is None or completion.goal_reached_at_s is None:
            return InvariantViolation(
                invariant_name="undone_timing",
                severity="error",
                details="Outcome is success_then_undone but timestamps are missing"
            )
        
        if completion.undone_at_s <= completion.goal_reached_at_s:
            return InvariantViolation(
                invariant_name="undone_timing",
                severity="error",
                details=f"undone_at_s must be > goal_reached_at_s"
            )
    return None
```

---

### Invariant 4: Progress Monotonicity (progress_monotonicity)

**Formal Statement**:
```
For all timeline, for all i, j where i < j:
  (windows[i..j] all have contribution = "advancing") implies (windows[j].progress >= windows[i].progress)
```

**Plain English**: In temporal windows where the arm contribution is advancing, the completion percentage must be monotonically non-decreasing. Progress cannot go backwards.

**Violation Conditions**:
- Consecutive advancing windows with progress[i+1] < progress[i]
- idle window with progress increase

**Implementation**:
```python
def check_progress_monotonicity(timeline: Timeline) -> Optional[InvariantViolation]:
    windows = timeline.windows
    
    for i in range(len(windows) - 1):
        current = windows[i]
        next_window = windows[i + 1]
        
        if current.contribution_type == "advancing":
            if next_window.progress < current.progress:
                return InvariantViolation(
                    invariant_name="progress_monotonicity",
                    severity="error",
                    details=f"Progress decreased during advancing phase"
                )
        
        if current.contribution_type == "idle":
            if next_window.progress > current.progress:
                return InvariantViolation(
                    invariant_name="idle_contribution_consistency",
                    severity="error",
                    details=f"Progress increased during idle phase"
                )
    
    return None
```

---

### Invariant 5: Time Bounds (time_bounds)

**Formal Statement**:
```
For all audit_report, for all timestamp t in report:
  0.0 <= t <= episode_length_s
```

**Plain English**: All event timestamps must fall within the episode duration. No events can occur before the episode starts or after it ends.

**Violation Conditions**:
- Any timestamp t where t < 0.0 OR t > episode_length_s

**Implementation**:
```python
def check_time_bounds(report: AuditReport) -> List[InvariantViolation]:
    violations = []
    episode_length = report.context.length_s
    
    # Check timeline windows
    for i, window in enumerate(report.timeline.windows):
        if not (0.0 <= window.start_s <= episode_length):
            violations.append(InvariantViolation(
                invariant_name="time_bounds",
                severity="error",
                details=f"Window start_s outside episode bounds"
            ))
        
        if not (0.0 <= window.end_s <= episode_length):
            violations.append(InvariantViolation(
                invariant_name="time_bounds",
                severity="error",
                details=f"Window end_s outside episode bounds"
            ))
    
    return violations if violations else None
```

---

## Invariant Verification Pipeline

All invariants MUST be checked before emitting audit reports:

```python
def verify_all_invariants(report: AuditReport) -> List[InvariantViolation]:
    violations = []
    
    violations.extend(filter(None, [
        check_outcome_vs_alignment(report),
        check_progress_vs_outcome(report),
        check_undone_timing(report),
        check_progress_monotonicity(report.timeline)
    ]))
    
    time_violations = check_time_bounds(report)
    if time_violations:
        violations.extend(time_violations)
    
    return violations
```

---

## Summary

These five core invariants provide formal guarantees of logical consistency:

1. **outcome_vs_alignment**: Success requires aligned goal
2. **progress_vs_outcome**: Failure/partial cannot reach 100% completion
3. **undone_timing**: Undo must occur after goal achievement
4. **progress_monotonicity**: Progress cannot decrease during active phases
5. **time_bounds**: All timestamps must fall within episode duration
