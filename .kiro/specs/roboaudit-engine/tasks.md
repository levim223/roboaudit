# Implementation Plan: RoboAudit Engine

## Overview

This implementation plan builds the RoboAudit data quality engine for robotics learning demonstrations. The engine follows a deterministic, format-agnostic architecture with formal invariant checking. Implementation proceeds in 5 phases: core infrastructure, invariant system, analysis components, property-based test suite, and visualization/batch processing.

## Tasks

- [x] 1. Set up project structure and core data models
  - Create Python package structure: `roboaudit/` with modules for `core/`, `parsers/`, `analysis/`, `reporting/`, `visualization/`
  - Define core data classes: `EpisodeMetadata`, `ExtractedEpisode`, `Frame`, `TemporalWindow`, `Timeline`, `TaskOutcome`, `AuditConfig`
  - Implement `AuditConfig` with default thresholds (jitter 2.0%, sensor desync 50ms, camera sync 16ms, hesitation 2.0s)
  - Create JSON schema validation for audit reports with `schema_version: "1.0.0"`
  - Set up `pyproject.toml` with dependencies: `av>=10.0.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`, `pandas>=2.0.0`, `hypothesis>=6.90.0`
  - _Requirements: 1.1, 1.3, 1.4, 12.1, 12.2_

- [x] 2. Implement Episode Reader and Format Detection
  - [x] 2.1 Create ParserPlugin abstract interface
    - Define `ParserPlugin` ABC with methods: `format_name`, `priority`, `matches_format`, `extract_videos`, `extract_telemetry`, `extract_metadata`
    - Implement `ParserPluginRegistry` for plugin registration and format matching
    - Add plugin priority resolution (highest priority wins when multiple plugins match)
    - _Requirements: 15.1, 15.2, 15.4_
  
  - [x] 2.2 Implement EpisodeReader with archive decompression
    - Write `EpisodeReader.read_episode()` with format detection and plugin delegation
    - Add archive decompression support for zip, tar, gzip formats
    - Implement missing component error reporting with descriptive messages
    - _Requirements: 1.1, 1.2, 1.5_
  
  - [x] 2.3 Implement video extraction with integer PTS sampling
    - Create `VideoExtractor` using PyAV with integer PTS sampling at 1.0s intervals
    - Extract frames with microsecond precision timestamps
    - Support multiple codecs: H264, H265, VP9, raw frames
    - Return `List[Frame]` with timestamp_us, pts, image (BGR), camera_id
    - _Requirements: 1.3, 1.6_
  
  - [x] 2.4 Implement telemetry extraction
    - Create `TelemetryExtractor` supporting CSV, JSON, Protocol Buffers, HDF5 formats
    - Parse telemetry into pandas DataFrame with timestamp column
    - Preserve original sampling rates and synchronization markers
    - _Requirements: 1.4, 1.7_
  
  - [x]* 2.5 Write unit tests for Episode Reader
    - Test archive decompression for each format (zip, tar, gzip)
    - Test missing component error reporting
    - Test codec support (one example per codec)
    - Test telemetry format support (one example per format)
    - _Requirements: 1.2, 1.5, 1.6, 1.7_

- [x] 3. Checkpoint - Verify episode ingestion
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Timebase Verifier
  - [x] 4.1 Create TimebaseVerifier for video frame jitter
    - Implement `check_video_timebase()` with 2.0% jitter threshold
    - Calculate frame interval variance: `abs(delta_t - expected_interval) / expected_interval >= 0.02`
    - Detect non-monotonic timestamp sequences (high severity error)
    - Calculate mean frame interval and standard deviation for each stream
    - Return `List[TimebaseIssue]` with issue_type, severity, time_range, affected_stream
    - _Requirements: 2.1, 2.3, 2.5_
  
  - [x] 4.2 Implement telemetry timebase checking
    - Implement `check_telemetry_timebase()` with 2.0% jitter threshold
    - Apply same jitter formula to consecutive telemetry samples
    - Calculate mean sampling interval and standard deviation for each channel
    - _Requirements: 2.2, 2.6_
  
  - [x] 4.3 Implement cross-modal synchronization check
    - Implement `check_cross_modal_sync()` with 50ms tolerance
    - Detect lag between video and telemetry timestamps
    - Flag sensor_desync warnings when `abs(video_ts - nearest_telemetry_ts) > 50ms`
    - _Requirements: 2.4_
  
  - [x]* 4.4 Write property test for frame jitter detection
    - **Property 6: Frame Jitter Detection**
    - **Validates: Requirements 2.1**
    - Generate frame sequences with controlled jitter using Hypothesis
    - Verify warnings flagged when variance exceeds 2.0%
  
  - [x]* 4.5 Write property test for telemetry jitter detection
    - **Property 7: Telemetry Jitter Detection**
    - **Validates: Requirements 2.2**
    - Generate telemetry with sampling jitter using Hypothesis
    - Verify warnings flagged when jitter exceeds 2.0%
  
  - [x]* 4.6 Write property test for non-monotonic detection
    - **Property 8: Non-Monotonic Timestamp Detection**
    - **Validates: Requirements 2.3**
    - Generate timestamp sequences with decreases or duplicates
    - Verify high-severity errors flagged

- [ ] 5. Implement Invariant Checker
  - [ ] 5.1 Create InvariantChecker with outcome-completion consistency
    - Implement `check_outcome_vs_completion()`: verify `(outcome == failure OR partial) → completion < 1.0`
    - Implement `check_progress_vs_outcome()`: verify `outcome == success → final_completion >= 0.95`
    - Return `Optional[InvariantViolation]` with invariant_name, severity, details
    - _Requirements: 5.1, 5.3_
  
  - [ ] 5.2 Implement temporal ordering invariants
    - Implement `check_undone_timing()`: verify `undone_at_s > goal_reached_at_s` for success_then_undone
    - Implement `check_time_past_end()`: verify all window timestamps <= episode_duration_s
    - Implement `check_progress_monotonicity()`: verify completion[i+1] >= completion[i] during advancing phases
    - _Requirements: 5.2, 4.5_
  
  - [ ] 5.3 Implement idle contribution consistency check
    - Implement `check_idle_contribution_consistency()`: verify idle windows have no completion increase
    - Detect outcome mismatches and report high-severity outcome_mismatch errors
    - _Requirements: 5.4, 5.5_
  
  - [ ]* 5.4 Write property test for outcome-completion consistency
    - **Property 18: Outcome-Completion Consistency**
    - **Validates: Requirements 5.1**
    - Generate timelines with failure/partial outcomes and completion >= 1.0
    - Verify invariant violations detected
  
  - [ ]* 5.5 Write property test for progress monotonicity
    - **Property 16: Progress Monotonicity During Advancement**
    - **Validates: Requirements 4.5**
    - Generate timelines with decreasing completion during advancing phases
    - Verify invariant violations detected
  
  - [ ]* 5.6 Write property test for undone temporal ordering
    - **Property 19: Undone Temporal Ordering**
    - **Validates: Requirements 5.2**
    - Generate success_then_undone outcomes with undone_at_s <= goal_reached_at_s
    - Verify invariant violations detected

- [ ] 6. Implement Audit Report Generator and Parser
  - [ ] 6.1 Create AuditReportGenerator
    - Implement `generate_report()` aggregating timeline, issues, mistakes, violations
    - Define `AuditReport` dataclass with schema_version, context, timeline, completion, goal_alignment, data_issues, operator_mistakes, quality_metrics
    - Implement `calculate_quality_score()` with weighted deductions for anomalies
    - Validate emitted reports against JSON schema before writing
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_
  
  - [ ] 6.2 Implement dual format export
    - Implement `export_json()` serializing AuditReport to JSON
    - Implement `export_markdown()` formatting report as human-readable markdown
    - _Requirements: 7.7_
  
  - [ ] 6.3 Implement AuditParser
    - Implement `parse_report()` parsing JSON strings into AuditReport objects
    - Implement `validate_schema()` with schema version checking
    - Return descriptive errors for invalid reports and unknown schema versions
    - _Requirements: 8.1, 8.2, 8.5, 8.6_
  
  - [ ] 6.4 Implement AuditPrettyPrinter
    - Implement `format_report()` serializing AuditReport back to valid JSON
    - _Requirements: 8.3_
  
  - [ ]* 6.5 Write property test for serialization round-trip
    - **Property 34: Report Serialization Round-Trip**
    - **Validates: Requirements 8.4**
    - Generate random AuditReport objects using Hypothesis
    - Verify parse → print → parse produces equivalent objects
  
  - [ ]* 6.6 Write property test for report structure completeness
    - **Property 28: Audit Report Structure Completeness**
    - **Validates: Requirements 7.1**
    - Generate valid episodes and complete audits
    - Verify all required sections present in emitted reports

- [ ] 7. Checkpoint - Verify core infrastructure and invariants
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement Action Phase Segmenter
  - [ ] 8.1 Create ActionPhaseSegmenter with phase classification
    - Implement `segment_episode()` classifying temporal windows into action phases
    - Classify phases: approach (gripper open + velocity toward object), grasp (open→closed transition), manipulate (closed + velocity > threshold), release (closed→open), idle (velocity < threshold for > 2.0s)
    - Assign arm attribution: left, right, both, none
    - Return `Timeline` with `List[TemporalWindow]`
    - _Requirements: 4.1, 4.2_
  
  - [ ] 8.2 Implement contribution classification
    - Implement `classify_contribution()` determining advancing, wasteful, or idle
    - Calculate completion percentage for each window (0.0 to 1.0)
    - Ensure `Timeline.validate_monotonicity()` verifies non-decreasing completion during advancing
    - _Requirements: 4.3, 4.4, 4.5_
  
  - [ ] 8.3 Implement operator hesitation detection
    - Detect Action_Phase persisting > 5.0s without completion progress
    - Flag operator_hesitation warnings
    - _Requirements: 4.7_
  
  - [ ]* 8.4 Write property test for temporal window classification completeness
    - **Property 15: Temporal Window Classification Completeness**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    - Generate episode data and segment into timeline
    - Verify all windows have action_phase, arm, contribution_type, completion in [0.0, 1.0]
  
  - [ ]* 8.5 Write property test for operator hesitation detection
    - **Property 17: Operator Hesitation Detection**
    - **Validates: Requirements 4.7**
    - Generate action phases with prolonged duration without progress
    - Verify operator_hesitation warnings flagged

- [ ] 9. Implement Grasp Anomaly Detector
  - [ ] 9.1 Create GraspAnomalyDetector with sensor-visual contradiction checking
    - Implement `detect_anomalies()` cross-referencing gripper state with visual evidence
    - Detect phantom grasp: `gripper_closed AND elevation_delta == 0` → high severity
    - Detect missed drop: `gripper_closed AND visual_object_falling` → high severity
    - Detect sensor mismatch: `gripper_force > 0 AND no_object_in_gripper_bbox` → medium severity
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [ ] 9.2 Implement object elevation measurement
    - Implement `measure_object_elevation()` using optical flow across frames
    - Calculate elevation delta for object tracking within gripper bounds
    - _Requirements: 3.5_
  
  - [ ] 9.3 Add camera evidence annotation
    - Annotate all grasp anomalies with Camera_Evidence references and frame numbers
    - _Requirements: 3.4_
  
  - [ ]* 9.4 Write property test for phantom grasp detection
    - **Property 11: Phantom Grasp Detection**
    - **Validates: Requirements 3.1**
    - Generate episodes with gripper_closed and elevation_delta == 0
    - Verify high-severity grasp_anomaly recorded
  
  - [ ]* 9.5 Write property test for object drop detection
    - **Property 13: Object Drop Detection**
    - **Validates: Requirements 3.3**
    - Generate episodes with visual object dropping and gripper_closed
    - Verify high-severity grasp_anomaly recorded
  
  - [ ]* 9.6 Write property test for anomaly evidence completeness
    - **Property 14: Anomaly Evidence Completeness**
    - **Validates: Requirements 3.4**
    - Generate grasp anomalies
    - Verify all include Camera_Evidence with frame numbers

- [ ] 10. Implement Multi-Camera Sync Checker
  - [ ] 10.1 Create CameraSyncChecker
    - Implement `check_timestamp_sync()` with 16ms tolerance
    - Verify frame timestamps stay synchronized across cameras
    - Flag camera_desync warnings when divergence > 16ms
    - _Requirements: 10.1, 10.2_
  
  - [ ] 10.2 Implement frame count consistency check
    - Implement `check_frame_counts()` with 2-frame tolerance
    - Flag camera_desync warnings when counts differ by > 2 frames
    - _Requirements: 10.3_
  
  - [ ] 10.3 Implement camera swap detection
    - Implement `detect_camera_swap()` via frame-to-frame feature consistency
    - Analyze viewpoint geometry for identifier swaps
    - Flag high-severity camera_swap errors
    - _Requirements: 10.4, 10.5_
  
  - [ ]* 10.4 Write property test for multi-camera timestamp synchronization
    - **Property 41: Multi-Camera Timestamp Synchronization**
    - **Validates: Requirements 10.1, 10.2**
    - Generate multi-camera episodes with desynchronized timestamps
    - Verify camera_desync warnings flagged when > 16ms

- [ ] 11. Implement Sensor Dropout Monitor
  - [ ] 11.1 Create SensorDropoutMonitor
    - Implement `detect_dropouts()` with 100ms gap threshold
    - Flag sensor_dropout warnings with affected channel, time range, duration, and percentage
    - _Requirements: 11.1, 11.5_
  
  - [ ] 11.2 Implement frozen sensor detection
    - Implement `detect_frozen_sensors()` for constant values > 1.0s during motion
    - Flag sensor_freeze warnings
    - _Requirements: 11.2_
  
  - [ ] 11.3 Implement outlier detection
    - Implement `detect_outliers()` validating against joint limits and force sensor ranges
    - Flag sensor_outlier warnings with affected samples
    - _Requirements: 11.3, 11.4_
  
  - [ ]* 11.4 Write property test for sensor dropout detection
    - **Property 44: Sensor Dropout Detection**
    - **Validates: Requirements 11.1**
    - Generate telemetry with gaps > 100ms
    - Verify sensor_dropout warnings flagged with correct time range
  
  - [ ]* 11.5 Write property test for sensor freeze detection
    - **Property 45: Sensor Freeze Detection**
    - **Validates: Requirements 11.2**
    - Generate telemetry with constant values during motion phases
    - Verify sensor_freeze warnings flagged

- [ ] 12. Implement Operator Mistake Classifier
  - [ ] 12.1 Create OperatorMistakeClassifier
    - Implement `classify_mistakes()` detecting drops, alignment struggles, hesitations, fumbles
    - Detect drops via grasp anomaly + visual evidence → high severity
    - Detect alignment struggles: approach angle error > 15° for > 2.0s → medium severity
    - Detect hesitations: idle phase > 2.0s during active task → low severity
    - Detect fumbles: trajectory oscillation > 3 reversals/s → medium severity
    - Annotate all mistakes with timestamp, severity, type, camera evidence
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [ ] 12.2 Implement mistake summary aggregation
    - Aggregate operator_mistakes into summary report with counts by severity and type
    - _Requirements: 6.6_
  
  - [ ]* 12.3 Write property test for drop mistake classification
    - **Property 23: Drop Mistake Classification**
    - **Validates: Requirements 6.1**
    - Generate episodes with visual object drops
    - Verify operator_mistakes entry with type "drop" and high severity
  
  - [ ]* 12.4 Write property test for operator mistake metadata completeness
    - **Property 27: Operator Mistake Metadata Completeness**
    - **Validates: Requirements 6.5**
    - Generate operator mistakes
    - Verify all include timestamp, severity, type, camera_evidence

- [ ] 13. Checkpoint - Verify analysis components
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Implement Configuration System
  - [ ] 14.1 Create configuration loading and validation
    - Implement configuration file parsing with jitter thresholds, timeout values, severity mappings
    - Validate configuration and report descriptive errors for invalid settings
    - Use documented defaults when no configuration provided
    - Support per-robot-type profiles (umi, franka, kinova)
    - _Requirements: 12.1, 12.2, 12.3, 12.5_
  
  - [ ] 14.2 Include configuration in audit report metadata
    - Add active configuration to audit report metadata section
    - _Requirements: 12.4_
  
  - [ ]* 14.3 Write property test for configuration application
    - **Property 48: Configuration Application**
    - **Validates: Requirements 12.1**
    - Generate valid configuration files with custom thresholds
    - Verify Audit_Engine applies specified values during processing
  
  - [ ]* 14.4 Write property test for invalid configuration error reporting
    - **Property 49: Invalid Configuration Error Reporting**
    - **Validates: Requirements 12.3**
    - Generate invalid configuration files
    - Verify descriptive errors reported

- [ ] 15. Implement Goal Alignment Scoring
  - [ ] 15.1 Create goal alignment calculator
    - Implement `calculate_goal_alignment()` scoring from 0.0 to 1.0
    - Apply weighted deductions for wasteful segments, operator mistakes, data anomalies
    - Enforce constraint: failure outcome → score <= 0.5
    - Include breakdown of deduction sources in audit report
    - Support custom scoring functions via configuration
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [ ]* 15.2 Write property test for goal alignment score range
    - **Property 55: Goal Alignment Score Range**
    - **Validates: Requirements 14.1**
    - Generate processed episodes
    - Verify goal_alignment score in [0.0, 1.0]
  
  - [ ]* 15.3 Write property test for failure outcome alignment constraint
    - **Property 57: Failure Outcome Alignment Constraint**
    - **Validates: Requirements 14.5**
    - Generate episodes with failure outcome
    - Verify goal_alignment score <= 0.5

- [ ] 16. Implement comprehensive property-based test suite
  - [ ] 16.1 Create Hypothesis strategies for core data types
    - Implement `st.episodes()` generating random episode structures
    - Implement `st.timelines()` generating temporal window sequences
    - Implement `st.audit_reports()` generating audit report objects
    - Implement `st.temporal_windows()` with phase/contribution constraints
    - Implement `st.telemetry_streams()` with controlled jitter/gaps
    - Configure Hypothesis settings: `max_examples=100`
  
  - [ ] 16.2 Write property tests for episode ingestion (Properties 1-5)
    - **Property 1: Episode Extraction Completeness** - Requirements 1.1
    - **Property 2: Archive Decompression and Extraction** - Requirements 1.2
    - **Property 3: Timestamp Precision Preservation** - Requirements 1.3
    - **Property 4: Telemetry Sampling Rate Preservation** - Requirements 1.4
    - **Property 5: Missing Component Error Reporting** - Requirements 1.5
  
  - [ ] 16.3 Write property tests for timebase verification (Properties 6-10)
    - Already written in task 4.4, 4.5, 4.6
    - **Property 9: Cross-Modal Desync Detection** - Requirements 2.4
    - **Property 10: Timebase Statistics Calculation** - Requirements 2.5, 2.6
  
  - [ ] 16.4 Write property tests for grasp anomaly detection (Properties 11-14)
    - Already written in task 9.4, 9.5, 9.6
    - **Property 12: Sensor-Visual Mismatch Detection** - Requirements 3.2
  
  - [ ] 16.5 Write property tests for action segmentation (Properties 15-17)
    - Already written in task 8.4, 8.5
  
  - [ ] 16.6 Write property tests for invariant checking (Properties 18-22)
    - Already written in task 5.4, 5.5, 5.6
    - **Property 20: Success Completion Threshold** - Requirements 5.3
    - **Property 21: Idle Window Completion Stasis** - Requirements 5.4
    - **Property 22: Outcome Contradiction Detection** - Requirements 5.5
  
  - [ ] 16.7 Write property tests for operator mistakes (Properties 23-27)
    - Already written in task 12.3, 12.4
    - **Property 24: Alignment Struggle Detection** - Requirements 6.2
    - **Property 25: Hesitation Mistake Detection** - Requirements 6.3
    - **Property 26: Fumble Detection** - Requirements 6.4
  
  - [ ] 16.8 Write property tests for audit reports (Properties 28-36)
    - Already written in task 6.5, 6.6
    - **Property 29: Error Report Generation** - Requirements 7.5
    - **Property 30: Report Schema Validation** - Requirements 7.6
    - **Property 31: Dual Format Export** - Requirements 7.7
    - **Property 32: Valid Report Parsing** - Requirements 8.1
    - **Property 33: Invalid Report Error Reporting** - Requirements 8.2
    - **Property 35: Required Field Validation** - Requirements 8.5
    - **Property 36: Schema Version Validation** - Requirements 8.6
  
  - [ ] 16.9 Write property tests for timeline visualization (Properties 37-40)
    - **Property 37: Timeline Board Generation** - Requirements 9.1
    - **Property 38: Timeline Board Structure** - Requirements 9.2
    - **Property 39: Contribution Type Color Coding** - Requirements 9.5
    - **Property 40: Progress Bar Rendering** - Requirements 9.6
  
  - [ ] 16.10 Write property tests for camera sync (Properties 41-43)
    - Already written in task 10.4
    - **Property 42: Camera Frame Count Consistency** - Requirements 10.3
    - **Property 43: Camera Swap Detection** - Requirements 10.4, 10.5
  
  - [ ] 16.11 Write property tests for sensor monitoring (Properties 44-47)
    - Already written in task 11.4, 11.5
    - **Property 46: Sensor Outlier Detection** - Requirements 11.3, 11.4
    - **Property 47: Dropout Duration Calculation** - Requirements 11.5
  
  - [ ] 16.12 Write property tests for configuration (Properties 48-50)
    - Already written in task 14.3, 14.4
    - **Property 50: Configuration Metadata Inclusion** - Requirements 12.4
  
  - [ ] 16.13 Write property tests for batch processing (Properties 51-54)
    - **Property 51: Batch Processing Completeness** - Requirements 13.1, 13.3
    - **Property 52: Parallel Processing Correctness** - Requirements 13.2
    - **Property 53: Batch Summary Generation** - Requirements 13.4
    - **Property 54: Batch Error Resilience** - Requirements 13.5
  
  - [ ] 16.14 Write property tests for goal alignment (Properties 55-57)
    - Already written in task 15.2, 15.3
    - **Property 56: Goal Alignment Penalty Application** - Requirements 14.2
  
  - [ ] 16.15 Write property tests for plugin architecture (Properties 58-60)
    - **Property 58: Plugin Registration and Selection** - Requirements 15.1, 15.2
    - **Property 59: Plugin Priority Resolution** - Requirements 15.4
    - **Property 60: Plugin Failure Handling** - Requirements 15.5

- [ ] 17. Checkpoint - Verify property-based test suite
  - Ensure all 60 property tests pass with 100 iterations each, ask the user if questions arise.

- [ ] 18. Implement Interactive Timeline Board
  - [ ] 18.1 Create TimelineBoardGenerator
    - Implement `generate_html()` producing self-contained HTML file
    - Create HTML5/Canvas multi-track layout: video thumbnails, telemetry plots, action phases, anomaly markers
    - Apply color-coded contribution types: green=advancing, yellow=wasteful, gray=idle
    - Render completion progress bar synchronized with timeline position
    - _Requirements: 9.1, 9.2, 9.5, 9.6_
  
  - [ ] 18.2 Implement interactive features
    - Add click-to-inspect anomaly details with frame references
    - Implement playback controls (play, pause, seek, speed adjustment)
    - Add zoom and pan on temporal axis
    - _Requirements: 9.3, 9.4, 9.7_
  
  - [ ] 18.3 Implement lightweight HTTP server
    - Implement `serve_board()` with Python HTTP server on configurable port
    - _Requirements: 9.1_
  
  - [ ]* 18.4 Write integration tests for timeline board
    - Test HTML generation with complete audit reports
    - Verify multi-track structure presence (video, telemetry, phases, anomalies)
    - Verify color coding and progress bar elements in HTML

- [ ] 19. Implement Batch Processor
  - [ ] 19.1 Create BatchProcessor with parallel processing
    - Implement `process_batch()` processing multiple episode subdirectories
    - Support parallel processing with configurable worker thread count
    - Generate individual audit reports for each episode
    - Provide progress reporting (episodes completed, estimated time remaining)
    - _Requirements: 13.1, 13.2, 13.3, 13.6_
  
  - [ ] 19.2 Implement aggregate summary generation
    - Implement `generate_summary()` with quality statistics across all episodes
    - Continue processing remaining episodes when individual episodes fail
    - Report failures in aggregate summary
    - _Requirements: 13.4, 13.5_
  
  - [ ]* 19.3 Write integration tests for batch processing
    - Test parallel processing with multiple episodes
    - Test error resilience with failing episodes
    - Verify aggregate summary generation

- [ ] 20. Implement standard format parser plugins
  - [ ] 20.1 Create example parser plugins
    - Implement CSV+MP4 parser plugin for common robotics format
    - Implement HDF5 parser plugin for lab-specific formats
    - Register plugins with appropriate priorities
    - _Requirements: 15.1, 15.2_
  
  - [ ] 20.2 Write plugin integration tests
    - Test format detection and plugin selection
    - Test priority resolution with multiple matching plugins
    - Test graceful failure handling

- [ ] 21. Create command-line interface and documentation
  - [ ] 21.1 Implement CLI with argparse
    - Add `audit` command for single episode processing
    - Add `batch` command for multiple episode processing
    - Add `visualize` command for timeline board generation
    - Add `validate` command for audit report validation
    - Support configuration file path argument
  
  - [ ] 21.2 Write end-to-end integration tests
    - Test complete audit workflow with real robotics fixture data
    - Test multi-camera synchronization with hardware recordings
    - Test batch processing with realistic dataset sizes
  
  - [ ] 21.3 Create user documentation
    - Write README with installation instructions
    - Document configuration file format and thresholds
    - Provide examples for each CLI command
    - Document parser plugin interface for custom formats

- [ ] 22. Final checkpoint - Complete system verification
  - Ensure all 60 property tests pass, all integration tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each property-based test uses Hypothesis with 100 iterations minimum
- Testing strategy combines property tests (universal invariants) with example tests (specific codec/format support)
- Property tests validate the 60 correctness properties defined in the design document
- All tasks reference specific requirements for traceability
- Checkpoints ensure incremental validation at phase boundaries
- The implementation follows deterministic rules (no ML models) for reproducibility
- Plugin architecture enables format extensibility without core engine changes

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "2.4"] },
    { "id": 3, "tasks": ["2.5", "4.1", "4.2", "4.3"] },
    { "id": 4, "tasks": ["4.4", "4.5", "4.6", "5.1", "5.2", "5.3"] },
    { "id": 5, "tasks": ["5.4", "5.5", "5.6", "6.1", "6.2"] },
    { "id": 6, "tasks": ["6.3", "6.4"] },
    { "id": 7, "tasks": ["6.5", "6.6", "8.1"] },
    { "id": 8, "tasks": ["8.2", "8.3", "9.1"] },
    { "id": 9, "tasks": ["8.4", "8.5", "9.2", "9.3", "10.1", "10.2"] },
    { "id": 10, "tasks": ["9.4", "9.5", "9.6", "10.3", "11.1", "11.2", "11.3"] },
    { "id": 11, "tasks": ["10.4", "11.4", "11.5", "12.1", "12.2"] },
    { "id": 12, "tasks": ["12.3", "12.4", "14.1", "14.2"] },
    { "id": 13, "tasks": ["14.3", "14.4", "15.1"] },
    { "id": 14, "tasks": ["15.2", "15.3", "16.1"] },
    { "id": 15, "tasks": ["16.2", "16.3", "16.4", "16.5", "16.6", "16.7", "16.8"] },
    { "id": 16, "tasks": ["16.9", "16.10", "16.11", "16.12", "16.13", "16.14", "16.15"] },
    { "id": 17, "tasks": ["18.1", "18.2"] },
    { "id": 18, "tasks": ["18.3", "18.4", "19.1"] },
    { "id": 19, "tasks": ["19.2", "19.3", "20.1"] },
    { "id": 20, "tasks": ["20.2", "21.1"] },
    { "id": 21, "tasks": ["21.2", "21.3"] }
  ]
}
```
