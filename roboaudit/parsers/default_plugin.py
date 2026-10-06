"""Standard directory format parser plugin for robotics episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from roboaudit.core.models import EpisodeMetadata, Frame
from roboaudit.parsers.base import ParserPlugin, default_registry
from roboaudit.parsers.telemetry import TelemetryExtractor
from roboaudit.parsers.video import VideoExtractor


class StandardEpisodePlugin(ParserPlugin):
    """Parser plugin for standardized robotics episode directories.
    
    Expected structure:
    episode_dir/
      ├── metadata.json (optional)
      ├── videos/ or camera_*.mp4
      └── telemetry/ or telemetry.csv/json
    """

    def __init__(self) -> None:
        self.video_extractor = VideoExtractor()
        self.telemetry_extractor = TelemetryExtractor()

    @property
    def format_name(self) -> str:
        return "standard"

    @property
    def priority(self) -> int:
        return 10

    def matches_format(self, episode_path: Path) -> bool:
        """Match if directory contains videos, telemetry, or metadata."""
        if not episode_path.is_dir():
            return False

        has_videos = (
            (episode_path / "videos").is_dir()
            or any(episode_path.glob("*.mp4"))
            or any(episode_path.glob("*.avi"))
        )
        has_telemetry = (
            (episode_path / "telemetry").is_dir()
            or any(episode_path.glob("telemetry.*"))
            or (episode_path / "state.csv").is_file()
        )
        has_metadata = (episode_path / "metadata.json").is_file()

        return bool(has_videos or has_telemetry or has_metadata)

    def extract_metadata(self, episode_path: Path) -> EpisodeMetadata:
        """Extract metadata from metadata.json or infer from files."""
        meta_file = episode_path / "metadata.json"
        if meta_file.is_file():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return EpisodeMetadata(
                    episode_id=data.get("episode_id", episode_path.name),
                    duration_s=float(data.get("duration_s", 0.0)),
                    recording_date=None,
                    robot_type=data.get("robot_type", "generic_robot"),
                    camera_count=int(data.get("camera_count", 1)),
                    telemetry_channels=data.get("telemetry_channels", []),
                    dataset=data.get("dataset", "default"),
                    instruction=data.get("instruction", "Execute task"),
                    task_outcome=data.get("task_outcome", "success"),
                )
            except Exception:
                pass

        # Infer metadata
        video_files = list(self._find_video_files(episode_path).values())
        return EpisodeMetadata(
            episode_id=episode_path.name,
            duration_s=0.0,
            recording_date=None,
            robot_type="generic_robot",
            camera_count=max(1, len(video_files)),
            telemetry_channels=[],
        )

    def extract_videos(
        self, episode_path: Path, sample_interval_s: float = 1.0
    ) -> Dict[str, List[Frame]]:
        """Extract video frames for all discovered camera streams."""
        video_files = self._find_video_files(episode_path)
        extractor = VideoExtractor(sample_interval_s=sample_interval_s)
        streams: Dict[str, List[Frame]] = {}

        for camera_id, vpath in video_files.items():
            streams[camera_id] = extractor.extract_frames(vpath, camera_id=camera_id)

        return streams

    def extract_telemetry(self, episode_path: Path) -> pd.DataFrame:
        """Extract telemetry data from telemetry file."""
        candidates = [
            episode_path / "telemetry.csv",
            episode_path / "telemetry.json",
            episode_path / "telemetry.parquet",
            episode_path / "state.csv",
            episode_path / "telemetry" / "data.csv",
            episode_path / "telemetry" / "data.json",
        ]
        for c in candidates:
            if c.is_file():
                return self.telemetry_extractor.extract(c)

        # Look for any .csv or .json in telemetry/ or root
        if (episode_path / "telemetry").is_dir():
            for p in (episode_path / "telemetry").iterdir():
                if p.is_file() and p.suffix.lower() in {".csv", ".json", ".parquet"}:
                    return self.telemetry_extractor.extract(p)

        # Return empty DataFrame with timestamp column
        return pd.DataFrame({"timestamp": [0.0]})

    def _find_video_files(self, episode_path: Path) -> Dict[str, Path]:
        """Find video files and map to camera IDs."""
        mapping: Dict[str, Path] = {}
        search_dirs = [episode_path / "videos", episode_path]
        for sdir in search_dirs:
            if not sdir.is_dir():
                continue
            for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi"):
                for vfile in sorted(sdir.glob(ext)):
                    cam_name = vfile.stem
                    if cam_name not in mapping:
                        mapping[cam_name] = vfile
        return mapping


# Register standard plugin into default registry
default_registry.register(StandardEpisodePlugin())
