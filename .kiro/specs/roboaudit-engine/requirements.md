# Requirements Document

## Introduction

RoboAudit is an open-source, format-agnostic data quality engine for robotics learning demonstrations. The system audits teleoperated demonstrations (including arms, UMI grippers, and multi-camera recordings) for operator mistakes, hardware faults, and logical contradictions. The engine ensures that robotics training datasets meet quality standards before they are used for imitation learning or behavioral cloning.

## Glossary

- **Episode**: A single demonstration recording containing synchronized video streams and telemetry data
- **Telemetry**: Time-series data from robotic sensors including joint positions, gripper states, and force measurements
- **Timebase_Jitter**: Temporal inconsistency in frame timestamps or telemetry sampling exceeding acceptable thresholds
- **Grasp_Anomaly**: Contradiction between visual evidence and gripper sensor state during manipulation
- **Action_Phase**: Classification of robot behavior into discrete stages (approach, grasp, manipulate, release, idle)
- **Timeline**: Temporal sequence of action phases with contribution classifications and completion progress
- **Audit_Engine**: The core system component that processes episodes and generates quality reports
- **Episode_Reader**: Component responsible for parsing and extracting data from episode archives
- **Invariant_Checker**: Component that verifies logical consistency rules across episode data
- **Camera_Evidence**: Timestamped video frames used to support anomaly detection
- **Contribution_Type**: Classification of robot activity as advancing, wasteful, or idle relative to task goals
- **Completion_Percentage**: Quantitative measure of task progress from 0.0 (start) to 1.0 (complete)
- **Task_Outcome**: Final result classification (success, failure, partial, success_then_undone)
- **Interactive_Timeline_Board**: Web-based visualization showing multi-track temporal analysis
- **Format_Agnostic_Parser**: Component capable of ingesting multiple demonstration file formats

## Requirements

### Requirement 1: Episode Data Ingestion

**User Story:** As a robotics researcher, I want to load episode data from various formats, so that I can audit demonstrations regardless of recording system.

#### Acceptance Criteria

1. WHEN a user provides an episode directory containing video files and telemetry files, THE Episode_Reader SHALL extract all video streams and telemetry channels
2. WHEN a user provides an episode archive (zip, tar, or compressed format), THE Episode_Reader SHALL decompress and extract the contents before processing
3. WHEN extracting video streams, THE Episode_Reader SHALL produce timestamped frames with microsecond precision
4. WHEN extracting telemetry data, THE Episode_Reader SHALL preserve original sampling rates and synchronization markers
5. WHEN the episode directory is missing required video or telemetry files, THE Episode_Reader SHALL return a descriptive error identifying the missing components
6. THE Episode_Reader SHALL support multiple video codecs (H264, H265, VP9, raw frames)
7. THE Episode_Reader SHALL support multiple telemetry formats (CSV, JSON, Protocol Buffers, HDF5)

### Requirement 2: Timebase Verification

**User Story:** As a robotics researcher, I want to detect temporal inconsistencies in my recordings, so that I can identify hardware or recording system failures.

#### Acceptance Criteria

1. WHEN consecutive frame timestamps vary by more than 2.0 percent of the expected frame interval, THE Audit_Engine SHALL flag a timebase_jitter warning with the affected frame range
2. WHEN follower-arm telemetry exhibits sampling jitter exceeding 2.0 percent, THE Audit_Engine SHALL flag a timebase_jitter warning with the affected time range
3. WHEN frame timestamps are non-monotonic, THE Audit_Engine SHALL flag a timebase_jitter error with high severity
4. WHEN telemetry timestamps lag visual timestamps by more than 50 milliseconds, THE Audit_Engine SHALL flag a sensor_desync warning
5. THE Audit_Engine SHALL calculate and report the mean frame interval and standard deviation for each video stream
6. THE Audit_Engine SHALL calculate and report the mean telemetry sampling interval and standard deviation for each sensor channel

### Requirement 3: Grasp Anomaly Detection

**User Story:** As a robotics researcher, I want to detect contradictions between visual and sensor data, so that I can identify sensor failures or mislabeled demonstrations.

#### Acceptance Criteria

1. WHEN gripper motor state registers closed AND visual object elevation delta is zero, THE Audit_Engine SHALL record a grasp_anomaly with timestamp and high severity
2. WHEN gripper force sensor indicates contact AND visual evidence shows no object within gripper bounds, THE Audit_Engine SHALL record a grasp_anomaly with medium severity
3. WHEN visual evidence shows object dropping AND gripper state remains closed, THE Audit_Engine SHALL record a grasp_anomaly with high severity
4. FOR ALL grasp_anomaly detections, THE Audit_Engine SHALL include Camera_Evidence references with frame numbers
5. THE Audit_Engine SHALL measure visual object elevation delta using optical flow or object tracking across consecutive frames

### Requirement 4: Action Phase Segmentation

**User Story:** As a robotics researcher, I want to understand the temporal structure of demonstrations, so that I can identify inefficient or wasteful segments.

#### Acceptance Criteria

1. WHEN processing an episode, THE Audit_Engine SHALL classify each temporal window into Action_Phase categories (approach, grasp, manipulate, release, idle)
2. FOR ALL temporal windows, THE Audit_Engine SHALL assign arm attribution (left, right, both, none)
3. FOR ALL temporal windows, THE Audit_Engine SHALL assign Contribution_Type (advancing, wasteful, idle)
4. FOR ALL temporal windows, THE Audit_Engine SHALL calculate Completion_Percentage from 0.0 to 1.0
5. THE Audit_Engine SHALL ensure that consecutive temporal windows maintain monotonically non-decreasing completion when Contribution_Type is advancing
6. THE Audit_Engine SHALL assemble temporal windows into a Timeline data structure with phase boundaries
7. WHEN an Action_Phase persists longer than 5.0 seconds without completion progress, THE Audit_Engine SHALL flag an operator_hesitation warning

### Requirement 5: Task Outcome Validation

**User Story:** As a robotics researcher, I want to verify that task outcomes match timeline progression, so that I can detect mislabeled demonstrations.

#### Acceptance Criteria

1. WHEN Task_Outcome is failure OR Task_Outcome is partial, THE Invariant_Checker SHALL reject any Timeline where final Completion_Percentage equals 1.0
2. WHEN Task_Outcome is success_then_undone, THE Invariant_Checker SHALL verify that undone_at_s strictly exceeds goal_reached_at_s
3. WHEN Task_Outcome is success, THE Invariant_Checker SHALL verify that final Completion_Percentage is at least 0.95
4. WHEN any temporal window has Contribution_Type idle, THE Invariant_Checker SHALL verify that Completion_Percentage does not increase within that window
5. IF Task_Outcome contradicts Timeline progression, THEN THE Audit_Engine SHALL report an outcome_mismatch error with high severity

### Requirement 6: Operator Mistake Detection

**User Story:** As a robotics researcher, I want to identify operator mistakes in teleoperated demonstrations, so that I can filter low-quality training data.

#### Acceptance Criteria

1. WHEN visual evidence shows an object dropping from the gripper, THE Audit_Engine SHALL create an operator_mistakes entry with type drop and high severity
2. WHEN gripper approach alignment error exceeds 15 degrees for more than 2.0 seconds, THE Audit_Engine SHALL create an operator_mistakes entry with type alignment_struggle and medium severity
3. WHEN Action_Phase is idle for more than 2.0 seconds during task execution, THE Audit_Engine SHALL create an operator_mistakes entry with type hesitation and low severity
4. WHEN arm trajectory exhibits oscillation exceeding 3 reversals per second, THE Audit_Engine SHALL create an operator_mistakes entry with type fumble and medium severity
5. FOR ALL operator_mistakes entries, THE Audit_Engine SHALL include timestamp, severity, mistake type, and Camera_Evidence references
6. THE Audit_Engine SHALL aggregate operator_mistakes into a summary report with counts by severity and type

### Requirement 7: Audit Report Generation

**User Story:** As a robotics researcher, I want structured audit results, so that I can programmatically filter and analyze demonstration quality.

#### Acceptance Criteria

1. WHEN an audit completes successfully, THE Audit_Engine SHALL emit a JSON audit report containing timeline, completion, goal_alignment, data_issues, and operator_mistakes sections
2. THE Audit_Engine SHALL include a schema_version field in all JSON audit reports for backward compatibility
3. THE Audit_Engine SHALL include episode metadata (duration, frame count, telemetry channel count, recording date) in the audit report
4. THE Audit_Engine SHALL calculate and include aggregate quality metrics (total anomalies, critical issues, warning count, quality score)
5. WHEN the audit encounters unrecoverable errors, THE Audit_Engine SHALL emit a partial audit report with an error section describing the failure
6. THE Audit_Engine SHALL validate all emitted JSON audit reports against the defined schema before writing to disk
7. THE Audit_Engine SHALL support exporting audit reports in both JSON and human-readable markdown formats

### Requirement 8: Audit Report Parsing and Validation

**User Story:** As a developer, I want to parse audit reports back into structured data, so that I can verify round-trip consistency.

#### Acceptance Criteria

1. WHEN a valid JSON audit report is provided, THE Audit_Parser SHALL parse it into an AuditReport data structure
2. WHEN an invalid JSON audit report is provided, THE Audit_Parser SHALL return a descriptive error indicating the validation failure
3. THE Audit_Pretty_Printer SHALL format AuditReport data structures back into valid JSON audit reports
4. FOR ALL valid AuditReport objects, parsing then printing then parsing SHALL produce an equivalent AuditReport object (round-trip property)
5. THE Audit_Parser SHALL validate that all required fields are present according to the schema version
6. THE Audit_Parser SHALL reject audit reports with unknown schema versions

### Requirement 9: Interactive Timeline Visualization

**User Story:** As a robotics researcher, I want to visually explore audit results, so that I can quickly identify problematic segments in demonstrations.

#### Acceptance Criteria

1. WHEN an audit completes, THE Audit_Engine SHALL generate an Interactive_Timeline_Board HTML file
2. THE Interactive_Timeline_Board SHALL display multiple synchronized tracks (video streams, telemetry channels, action phases, anomalies)
3. WHEN a user clicks on an anomaly marker in the Interactive_Timeline_Board, THE Interactive_Timeline_Board SHALL display anomaly details and jump to the corresponding timestamp
4. THE Interactive_Timeline_Board SHALL support playback controls (play, pause, seek, speed adjustment)
5. THE Interactive_Timeline_Board SHALL highlight temporal segments by Contribution_Type using color coding (advancing=green, wasteful=yellow, idle=gray)
6. THE Interactive_Timeline_Board SHALL overlay Completion_Percentage as a progress bar synchronized with timeline position
7. THE Interactive_Timeline_Board SHALL support zooming and panning across the temporal axis

### Requirement 10: Multi-Camera Synchronization Verification

**User Story:** As a robotics researcher, I want to detect camera desynchronization, so that I can identify recording system failures.

#### Acceptance Criteria

1. WHEN multiple camera streams are present, THE Audit_Engine SHALL verify that frame timestamps across cameras remain synchronized within 16 milliseconds
2. WHEN camera frame timestamps diverge by more than 16 milliseconds, THE Audit_Engine SHALL flag a camera_desync warning with the affected camera pair
3. WHEN camera stream frame counts differ by more than 2 frames, THE Audit_Engine SHALL flag a camera_desync warning
4. WHEN camera stream identifiers swap during an episode, THE Audit_Engine SHALL flag a camera_swap error with high severity
5. THE Audit_Engine SHALL detect camera swapping by analyzing frame-to-frame feature consistency and viewpoint geometry

### Requirement 11: Sensor Dropout Detection

**User Story:** As a robotics researcher, I want to detect sensor data dropouts, so that I can identify hardware communication failures.

#### Acceptance Criteria

1. WHEN telemetry channels exhibit gaps exceeding 100 milliseconds, THE Audit_Engine SHALL flag a sensor_dropout warning with the affected channel and time range
2. WHEN telemetry values remain constant for more than 1.0 second during motion phases, THE Audit_Engine SHALL flag a sensor_freeze warning
3. WHEN telemetry values exceed physically plausible ranges, THE Audit_Engine SHALL flag a sensor_outlier warning with the affected samples
4. THE Audit_Engine SHALL validate telemetry values against configured joint limits and force sensor ranges
5. FOR ALL sensor_dropout warnings, THE Audit_Engine SHALL calculate the total duration and percentage of episode affected

### Requirement 12: Configurable Quality Thresholds

**User Story:** As a robotics researcher, I want to customize quality thresholds, so that I can adapt the engine to different robot platforms and task requirements.

#### Acceptance Criteria

1. THE Audit_Engine SHALL accept a configuration file specifying jitter thresholds, timeout values, and severity mappings
2. WHEN no configuration file is provided, THE Audit_Engine SHALL use documented default threshold values
3. THE Audit_Engine SHALL validate configuration files and report descriptive errors for invalid settings
4. THE Audit_Engine SHALL include the active configuration in the audit report metadata section
5. THE Audit_Engine SHALL support per-robot-type configuration profiles (arm_type: umi, franka, kinova)

### Requirement 13: Batch Episode Processing

**User Story:** As a robotics researcher, I want to audit multiple episodes in parallel, so that I can efficiently process large datasets.

#### Acceptance Criteria

1. WHEN a user provides a directory containing multiple episode subdirectories, THE Audit_Engine SHALL process all episodes
2. THE Audit_Engine SHALL support parallel processing of independent episodes using configurable worker thread count
3. WHEN batch processing, THE Audit_Engine SHALL generate individual audit reports for each episode
4. WHEN batch processing, THE Audit_Engine SHALL generate an aggregate summary report with quality statistics across all episodes
5. WHEN an individual episode fails to process, THE Audit_Engine SHALL continue processing remaining episodes and report the failure in the aggregate summary
6. THE Audit_Engine SHALL provide progress reporting during batch processing (episodes completed, estimated time remaining)

### Requirement 14: Goal Alignment Scoring

**User Story:** As a robotics researcher, I want to quantify how well demonstrations align with task goals, so that I can rank episode quality.

#### Acceptance Criteria

1. WHEN processing an episode, THE Audit_Engine SHALL calculate a goal_alignment score from 0.0 (complete misalignment) to 1.0 (perfect alignment)
2. THE goal_alignment score SHALL penalize wasteful segments, operator mistakes, and data anomalies using weighted deductions
3. THE Audit_Engine SHALL include goal_alignment in the audit report with a breakdown of deduction sources
4. THE Audit_Engine SHALL support custom goal_alignment scoring functions via configuration
5. WHEN Task_Outcome is failure, THE goal_alignment score SHALL not exceed 0.5

### Requirement 15: Format Plugin Architecture

**User Story:** As a developer, I want to extend the engine with custom format parsers, so that I can support proprietary recording formats.

#### Acceptance Criteria

1. THE Episode_Reader SHALL support registering format-specific parser plugins
2. WHEN a parser plugin is registered, THE Episode_Reader SHALL use it for episodes matching the plugin's format signature
3. THE Episode_Reader SHALL provide a documented parser plugin interface with methods for video extraction, telemetry extraction, and metadata retrieval
4. WHEN multiple parser plugins match an episode format, THE Episode_Reader SHALL use the plugin with the highest priority score
5. THE Episode_Reader SHALL gracefully handle parser plugin failures and report descriptive errors

---

## Implementation Notes

This requirements document prioritizes testability and logical consistency. Each acceptance criterion maps to verifiable properties that can be tested using property-based testing frameworks. The parser/pretty-printer round-trip requirement ensures data integrity, and the invariant checking requirements ensure that audit results are logically consistent with task outcomes.
