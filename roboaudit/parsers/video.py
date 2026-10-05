"""Video extraction with integer PTS sampling at exact intervals."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from roboaudit.core.models import Frame


class VideoExtractor:
    """Extracts timestamped video frames using integer PTS to eliminate rounding drift."""

    def __init__(self, sample_interval_s: float = 1.0) -> None:
        """Initialize VideoExtractor.
        
        Args:
            sample_interval_s: Interval in seconds between sampled frames (default: 1.0s).
        """
        self.sample_interval_s = float(sample_interval_s)

    def extract_frames(self, video_path: Path, camera_id: str = "camera_0") -> List[Frame]:
        """Extract frames from a video file sampled by exact integer PTS.
        
        Args:
            video_path: Path to video file (.mp4, .mov, .mkv, .avi, etc.).
            camera_id: Identifier name for this camera view.
            
        Returns:
            List of Frame instances.
            
        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video file cannot be opened or contains no video stream.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            return self._extract_with_pyav(video_path, camera_id)
        except Exception:
            return self._extract_with_opencv(video_path, camera_id)

    def _extract_with_pyav(self, video_path: Path, camera_id: str) -> List[Frame]:
        """Extract frames using PyAV with exact integer PTS."""
        import av

        container = av.open(str(video_path))
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is None:
            container.close()
            raise ValueError(f"No video stream found in {video_path}")

        time_base = float(video_stream.time_base) if video_stream.time_base else 1.0 / 30.0
        pts_interval = int(round(self.sample_interval_s / time_base)) if time_base > 0 else 30
        if pts_interval <= 0:
            pts_interval = 1

        frames: List[Frame] = []
        next_target_pts: Optional[int] = None

        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue

                if next_target_pts is None:
                    next_target_pts = frame.pts

                if frame.pts >= next_target_pts:
                    # Convert to BGR numpy array
                    img = frame.to_ndarray(format="bgr24")
                    # Microsecond timestamp
                    timestamp_us = int(round(float(frame.pts) * time_base * 1_000_000))

                    frames.append(
                        Frame(
                            timestamp_us=timestamp_us,
                            pts=int(frame.pts),
                            image=img,
                            camera_id=camera_id,
                        )
                    )
                    next_target_pts += pts_interval

        container.close()
        return frames

    def _extract_with_opencv(self, video_path: Path, camera_id: str) -> List[Frame]:
        """Fallback extraction using OpenCV VideoCapture."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = 30.0

        frame_step = max(1, int(round(self.sample_interval_s * fps)))
        frames: List[Frame] = []
        frame_idx = 0

        while True:
            ret, image = cap.read()
            if not ret or image is None:
                break

            if frame_idx % frame_step == 0:
                time_s = frame_idx / fps
                timestamp_us = int(round(time_s * 1_000_000))
                frames.append(
                    Frame(
                        timestamp_us=timestamp_us,
                        pts=frame_idx,
                        image=image,
                        camera_id=camera_id,
                    )
                )
            frame_idx += 1

        cap.release()
        return frames
