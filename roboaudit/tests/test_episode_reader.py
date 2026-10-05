"""Unit tests for EpisodeReader, ParserPluginRegistry, and telemetry extraction."""

from __future__ import annotations

import json
import tarfile
import tempfile
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from roboaudit.core.models import EpisodeMetadata, ExtractedEpisode
from roboaudit.parsers.base import ParserPlugin, ParserPluginRegistry
from roboaudit.parsers.default_plugin import StandardEpisodePlugin
from roboaudit.parsers.reader import EpisodeReader
from roboaudit.parsers.telemetry import TelemetryExtractor
from roboaudit.parsers.video import VideoExtractor


def create_dummy_video(path: Path, num_frames: int = 30, fps: int = 30) -> None:
    """Helper to create a small valid MP4 video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for i in range(num_frames):
        frame = np.full((64, 64, 3), i * 5, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestParserPluginRegistry:
    """Tests for ParserPlugin registration and priority resolution."""

    def test_registration_and_lookup(self):
        registry = ParserPluginRegistry()
        plugin = StandardEpisodePlugin()
        registry.register(plugin)

        assert registry.get_plugin("standard") is plugin
        assert "standard" in registry.list_plugins()

    def test_priority_resolution(self, tmp_path):
        class LowPriorityPlugin(ParserPlugin):
            @property
            def format_name(self) -> str:
                return "low"
            @property
            def priority(self) -> int:
                return 5
            def matches_format(self, p: Path) -> bool:
                return True
            def extract_metadata(self, p: Path) -> EpisodeMetadata:
                return EpisodeMetadata("low", 1.0, None, "bot", 1, [])
            def extract_videos(self, p: Path, sample_interval_s: float = 1.0):
                return {}
            def extract_telemetry(self, p: Path):
                return pd.DataFrame({"timestamp": [0.0]})

        class HighPriorityPlugin(ParserPlugin):
            @property
            def format_name(self) -> str:
                return "high"
            @property
            def priority(self) -> int:
                return 50
            def matches_format(self, p: Path) -> bool:
                return True
            def extract_metadata(self, p: Path) -> EpisodeMetadata:
                return EpisodeMetadata("high", 1.0, None, "bot", 1, [])
            def extract_videos(self, p: Path, sample_interval_s: float = 1.0):
                return {}
            def extract_telemetry(self, p: Path):
                return pd.DataFrame({"timestamp": [0.0]})

        registry = ParserPluginRegistry()
        registry.register(LowPriorityPlugin())
        registry.register(HighPriorityPlugin())

        matched = registry.find_matching_plugin(tmp_path)
        assert matched is not None
        assert matched.format_name == "high"


class TestTelemetryExtractor:
    """Tests for TelemetryExtractor on CSV and JSON formats."""

    def test_extract_csv_telemetry(self, tmp_path):
        csv_file = tmp_path / "telemetry.csv"
        csv_file.write_text("timestamp,joint_0,joint_1,gripper\n0.0,0.1,0.2,0.0\n0.1,0.15,0.25,1.0\n")

        extractor = TelemetryExtractor()
        df = extractor.extract(csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert df["gripper"].iloc[1] == 1.0

    def test_extract_json_records(self, tmp_path):
        json_file = tmp_path / "telemetry.json"
        data = [
            {"time_s": 0.0, "gripper": 0.0},
            {"time_s": 0.5, "gripper": 1.0},
        ]
        json_file.write_text(json.dumps(data))

        extractor = TelemetryExtractor()
        df = extractor.extract(json_file)

        assert len(df) == 2
        assert "timestamp" in df.columns
        assert df["timestamp"].iloc[1] == 0.5

    def test_nonexistent_file_raises_error(self, tmp_path):
        extractor = TelemetryExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract(tmp_path / "nonexistent.csv")


class TestEpisodeReader:
    """Tests for EpisodeReader directory ingestion and archive decompression."""

    @pytest.fixture
    def standard_episode_dir(self, tmp_path) -> Path:
        """Create a mock standard episode directory."""
        ep_dir = tmp_path / "test_episode_01"
        ep_dir.mkdir()

        # Metadata
        meta = {
            "episode_id": "test_episode_01",
            "duration_s": 2.0,
            "robot_type": "teleop_arm",
            "camera_count": 1,
            "telemetry_channels": ["joint_0", "gripper"],
        }
        (ep_dir / "metadata.json").write_text(json.dumps(meta))

        # Videos
        video_dir = ep_dir / "videos"
        video_dir.mkdir()
        create_dummy_video(video_dir / "front_camera.mp4", num_frames=30, fps=30)

        # Telemetry
        telemetry_csv = ep_dir / "telemetry.csv"
        telemetry_csv.write_text("timestamp,joint_0,gripper\n0.0,0.0,0.0\n1.0,0.5,1.0\n")

        return ep_dir

    def test_read_standard_directory(self, standard_episode_dir):
        reader = EpisodeReader()
        episode = reader.read_episode(standard_episode_dir)

        assert isinstance(episode, ExtractedEpisode)
        assert episode.metadata.episode_id == "test_episode_01"
        assert "front_camera" in episode.video_streams
        assert len(episode.video_streams["front_camera"]) > 0
        assert not episode.telemetry.empty

    def test_read_zip_archive(self, standard_episode_dir, tmp_path):
        zip_path = tmp_path / "episode.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for file_path in standard_episode_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(standard_episode_dir.parent)
                    zf.write(file_path, arcname)

        reader = EpisodeReader()
        episode = reader.read_episode(zip_path)

        assert episode.metadata.episode_id == "test_episode_01"
        assert "front_camera" in episode.video_streams

    def test_read_tar_gz_archive(self, standard_episode_dir, tmp_path):
        tar_path = tmp_path / "episode.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for file_path in standard_episode_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(standard_episode_dir.parent)
                    tf.add(file_path, arcname=str(arcname))

        reader = EpisodeReader()
        episode = reader.read_episode(tar_path)

        assert episode.metadata.episode_id == "test_episode_01"
        assert "front_camera" in episode.video_streams

    def test_missing_directory_raises_error(self, tmp_path):
        reader = EpisodeReader()
        with pytest.raises(FileNotFoundError):
            reader.read_episode(tmp_path / "does_not_exist")

    def test_unmatched_empty_directory_raises_value_error(self, tmp_path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        reader = EpisodeReader()
        with pytest.raises(ValueError, match="No parser plugin matched"):
            reader.read_episode(empty_dir)
