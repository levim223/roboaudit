"""EpisodeReader with archive decompression and format plugin delegation."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Union

from roboaudit.core.models import ExtractedEpisode
from roboaudit.parsers.base import ParserPluginRegistry, default_registry
import roboaudit.parsers.default_plugin  # ensure standard plugin is loaded


class EpisodeReader:
    """Reads and parses robotics demonstration episodes from directories or archives."""

    ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".tar.gz", ".bz2", ".tar.bz2"}

    def __init__(self, registry: Optional[ParserPluginRegistry] = None) -> None:
        """Initialize EpisodeReader with parser plugin registry."""
        self.registry = registry or default_registry

    def read_episode(
        self,
        episode_path: Union[str, Path],
        sample_interval_s: float = 1.0,
    ) -> ExtractedEpisode:
        """Read and extract an episode from a directory or compressed archive.
        
        Args:
            episode_path: Path to directory or archive (.zip, .tar, .tar.gz).
            sample_interval_s: Sampling interval for integer PTS video extraction.
            
        Returns:
            ExtractedEpisode containing metadata, video streams, and telemetry DataFrame.
            
        Raises:
            FileNotFoundError: If episode path does not exist.
            ValueError: If format is unrecognized or required components are missing.
        """
        path = Path(episode_path)
        if not path.exists():
            raise FileNotFoundError(f"Episode path does not exist: {path}")

        temp_dir: Optional[Path] = None
        working_dir = path

        try:
            if path.is_file():
                if self._is_archive(path):
                    temp_dir = Path(tempfile.mkdtemp(prefix="roboaudit_ep_"))
                    self._extract_archive(path, temp_dir)
                    # Check if extracted into a single subfolder
                    subitems = [p for p in temp_dir.iterdir() if not p.name.startswith(".")]
                    if len(subitems) == 1 and subitems[0].is_dir():
                        working_dir = subitems[0]
                    else:
                        working_dir = temp_dir
                else:
                    raise ValueError(f"File is not a supported episode archive: {path}")

            plugin = self.registry.find_matching_plugin(working_dir)
            if plugin is None:
                raise ValueError(
                    f"No parser plugin matched the episode at: {path}. "
                    f"Expected either a videos directory, telemetry files (CSV/JSON/Parquet), or metadata.json."
                )

            metadata = plugin.extract_metadata(working_dir)
            video_streams = plugin.extract_videos(
                working_dir, sample_interval_s=sample_interval_s
            )
            telemetry = plugin.extract_telemetry(working_dir)

            # Update duration if metadata lacked it and video frames exist
            if metadata.duration_s <= 0.0 and video_streams:
                max_us = max(
                    (frames[-1].timestamp_us for frames in video_streams.values() if frames),
                    default=0,
                )
                if max_us > 0:
                    metadata.duration_s = max_us / 1_000_000.0

            return ExtractedEpisode(
                metadata=metadata,
                video_streams=video_streams,
                telemetry=telemetry,
            )

        finally:
            if temp_dir is not None and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    # Alias for convenience
    read = read_episode

    def _is_archive(self, path: Path) -> bool:
        """Check if path is a supported compressed archive."""
        name = path.name.lower()
        return any(name.endswith(ext) for ext in self.ARCHIVE_EXTENSIONS)

    def _extract_archive(self, archive_path: Path, target_dir: Path) -> None:
        """Safely extract zip or tar archive preventing path traversal."""
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    target = (target_dir / member.filename).resolve()
                    if not str(target).startswith(str(target_dir.resolve())):
                        raise ValueError(f"Unsafe archive member path: {member.filename}")
                zf.extractall(target_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tf:
                for member in tf.getmembers():
                    target = (target_dir / member.name).resolve()
                    if not str(target).startswith(str(target_dir.resolve())):
                        raise ValueError(f"Unsafe archive member path: {member.name}")
                if hasattr(tarfile, "data_filter"):
                    tf.extractall(target_dir, filter="data")
                else:
                    tf.extractall(target_dir)
        else:
            raise ValueError(f"Unsupported or corrupted archive file: {archive_path}")
