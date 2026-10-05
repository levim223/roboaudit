---
inclusion: always
---

# Robotics Data Standards for RoboAudit Engine

This document defines the standardized data formats, directory structures, and capture thresholds for robotics demonstration episodes processed by the RoboAudit engine.

## Episode Directory Structure

All robotics episodes MUST follow this standardized directory layout:

`
episode_id/
├── videos/
│   ├── front_camera.mp4      # Front-view camera (H.264/H.265)
│   ├── wrist_camera.mp4      # Wrist-mounted camera
│   └── side_camera.mp4       # Side-view camera (optional)
├── telemetry/
│   ├── joint_states.csv      # Joint angles, velocities, torques
│   ├── gripper_states.csv    # Gripper position (0.0=open, 1.0=closed)
│   └── ee_poses.json         # End-effector poses (position + quaternion)
└── metadata.json             # Episode metadata (duration, robot type, task)
`

### Video Stream Standards

- **Encoding**: H.264 or H.265 codec in MP4 container; raw frames (PNG/JPEG sequence) also supported
- **Multi-Camera Setup**: Minimum 1 camera (front), recommended 3 cameras (front, wrist, side)
- **Resolution**: Minimum 640x480, recommended 1280x720 or higher
- **Frame Rate**: 30 FPS standard; 60 FPS for high-speed manipulation tasks
- **Synchronization**: All camera streams MUST have synchronized timestamps (within 16ms tolerance)

### Telemetry File Standards

- **Formats Supported**: CSV, JSON, Parquet, HDF5, Protocol Buffers
- **Required Fields**:
  - 	imestamp_us (int64): Microsecond-precision Unix timestamp
  - joint_positions (float array): Joint angles in radians
  - gripper_state (float): Gripper openness [0.0=fully open, 1.0=fully closed]
  - e_position (float[3]): End-effector position [x, y, z] in meters
  - e_orientation (float[4]): End-effector orientation as quaternion [x, y, z, w]
- **Optional Fields**:
  - joint_velocities, joint_torques, gripper_force, contact_forces
- **Sampling Rate**: Minimum 10 Hz, recommended 30 Hz or higher

## Frame Extraction Standards

### Integer PTS Sampling (CRITICAL)

Frame extraction MUST use integer Presentation Time Stamps (PTS) via PyAV to ensure deterministic, reproducible sampling:

`python
import av

container = av.open(video_path)
stream = container.streams.video[0]

# Extract frames at exact 1.0s intervals using integer PTS
target_pts = 0
pts_per_second = stream.time_base.denominator / stream.time_base.numerator

for frame in container.decode(stream):
    if frame.pts >= target_pts:
        timestamp_us = int(frame.pts * stream.time_base * 1_000_000)
        # Process frame with microsecond precision timestamp
        target_pts += pts_per_second  # Advance by 1.0s
`

**Rules:**
- ✅ **DO**: Use integer PTS arithmetic (rame.pts >= target_pts)
- ✅ **DO**: Sample at uniform 1.0s intervals for consistency
- ✅ **DO**: Preserve microsecond precision in timestamps (int(pts * time_base * 1_000_000))
- ❌ **DO NOT**: Use floating-point timestamp comparisons (causes rounding drift)
- ❌ **DO NOT**: Drop or interpolate frames based on approximate timestamps
- ❌ **DO NOT**: Rely on frame indices without validating PTS monotonicity

## Deterministic Hardware & Capture Thresholds

### Frame Jitter Detection

**Definition**: Temporal inconsistency in consecutive frame timestamps exceeding acceptable variance.

**Threshold**: 
`python
expected_interval = 1.0 / fps  # seconds
actual_interval = frame[i+1].timestamp - frame[i].timestamp
jitter_ratio = abs(actual_interval - expected_interval) / expected_interval

if jitter_ratio >= 0.02:  # 2.0%
    flag_issue("timebase_jitter", severity="warning")
`

**Triggers**:
- Frame interval variance ≥ 2.0% → 	imebase_jitter warning
- Non-monotonic timestamps (decrease or duplicate) → 	imebase_jitter error (high severity)

### Sped-Up Recording Detection

**Definition**: Recording played back faster than real-time during capture, causing compressed temporal dynamics.

**Detection Logic**:
`python
sample_jitter = calculate_jitter(telemetry_timestamps)
follower_lag = cross_correlate(leader_arm, follower_arm).lag_frames

if sample_jitter >= 0.02 and follower_lag <= 3.4:
    flag_issue("sped_up_playback", severity="error")
`

**Indicators**:
- Telemetry sampling jitter ≥ 2.0%
- AND follower arm lag compressed to ≤ 3.4 frames (normally 5-8 frames in real-time teleoperation)

### Sensor Desynchronization

**Definition**: Temporal misalignment between video frames and telemetry samples.

**Threshold**:
`python
closest_telemetry_ts = find_nearest_timestamp(telemetry, video_frame.timestamp)
desync_ms = abs(video_frame.timestamp - closest_telemetry_ts) * 1000

if desync_ms > 50.0:  # milliseconds
    flag_issue("sensor_desync", severity="warning")
`

**Triggers**:
- Video-telemetry timestamp discrepancy > 50ms → sensor_desync warning

### Grasp Anomaly Detection

**Definition**: Contradiction between gripper sensor state and visual evidence of object manipulation.

**Detection Rules**:

1. **Phantom Grasp** (High Severity):
`python
if gripper_state >= 0.8 and object_elevation_delta <= 0.01:  # closed + no lift
    flag_issue("grasp_anomaly", type="phantom_grasp", severity="high")
`

2. **Missed Drop** (High Severity):
`python
if gripper_state >= 0.8 and detect_object_falling(video_frames):  # closed + falling
    flag_issue("grasp_anomaly", type="missed_drop", severity="high")
`

3. **Sensor Mismatch** (Medium Severity):
`python
if gripper_force > 0.5 and not object_in_gripper_bbox(video_frame):  # force + no object
    flag_issue("grasp_anomaly", type="sensor_mismatch", severity="medium")
`

**Object Elevation Measurement**:
- Use optical flow or object tracking across consecutive frames
- Calculate z-axis displacement in meters: delta_z = object_pos[t+1].z - object_pos[t].z
- Minimum positive elevation threshold: 0.01m (1cm) to confirm successful grasp

### Camera Synchronization Verification

**Multi-Camera Timestamp Tolerance**:
`python
for cam_pair in itertools.combinations(cameras, 2):
    ts_diff_ms = abs(cam_pair[0].frame.timestamp - cam_pair[1].frame.timestamp) * 1000
    
    if ts_diff_ms > 16.0:  # milliseconds (approximately 1 frame at 60 FPS)
        flag_issue("camera_desync", cameras=cam_pair, severity="warning")
`

**Frame Count Consistency**:
`python
frame_counts = {cam_id: len(frames) for cam_id, frames in cameras.items()}
max_diff = max(frame_counts.values()) - min(frame_counts.values())

if max_diff > 2:  # frames
    flag_issue("camera_desync", type="frame_count_mismatch", severity="warning")
`

### Sensor Dropout Detection

**Gap Threshold**:
`python
for i in range(len(telemetry) - 1):
    gap_ms = (telemetry[i+1].timestamp - telemetry[i].timestamp) * 1000
    
    if gap_ms > 100.0:  # milliseconds
        flag_issue("sensor_dropout", channel=telemetry.channel, 
                   duration_ms=gap_ms, severity="warning")
`

**Frozen Sensor Detection**:
`python
window_duration = 1.0  # seconds
if is_motion_phase(window) and telemetry_constant(window, duration=window_duration):
    flag_issue("sensor_freeze", channel=telemetry.channel, severity="warning")
`

## Data Quality Compliance

All episodes processed by RoboAudit MUST:
1. ✅ Contain at least one video stream with valid timestamps
2. ✅ Contain synchronized telemetry with microsecond-precision timestamps
3. ✅ Have monotonically increasing timestamps (no duplicates or reversals)
4. ✅ Meet minimum sampling rates (video ≥10 FPS, telemetry ≥10 Hz)
5. ✅ Include gripper state telemetry for grasp analysis
6. ✅ Provide episode metadata (duration, robot type, task instruction)

Episodes failing these requirements will generate audit reports with high-severity errors in the data_issues section.
