"""ParserPlugin abstract base class and registry for format-agnostic episode ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type

from roboaudit.core.models import EpisodeMetadata, Frame
import pandas as pd


class ParserPlugin(ABC):
    """Abstract base class for format-specific episode parsers.
    
    Subclasses implement format detection and extraction for specific
    robot demonstration datasets (e.g., standard directories, LeRobot, UMI, ROS bags).
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Unique identifier name for this format (e.g. 'standard', 'lerobot')."""
        pass

    @property
    def priority(self) -> int:
        """Priority score when multiple plugins match a format. Higher wins."""
        return 10

    @abstractmethod
    def matches_format(self, episode_path: Path) -> bool:
        """Determine whether the specified path matches this format plugin.
        
        Args:
            episode_path: Path to the episode directory.
            
        Returns:
            True if this plugin can parse the episode, False otherwise.
        """
        pass

    @abstractmethod
    def extract_metadata(self, episode_path: Path) -> EpisodeMetadata:
        """Extract metadata describing the episode.
        
        Args:
            episode_path: Path to the episode directory.
            
        Returns:
            EpisodeMetadata instance.
        """
        pass

    @abstractmethod
    def extract_videos(self, episode_path: Path, sample_interval_s: float = 1.0) -> Dict[str, List[Frame]]:
        """Extract video frames sampled by exact integer PTS.
        
        Args:
            episode_path: Path to the episode directory.
            sample_interval_s: Target interval between sampled frames in seconds.
            
        Returns:
            Dictionary mapping camera_id to list of Frame instances.
        """
        pass

    @abstractmethod
    def extract_telemetry(self, episode_path: Path) -> pd.DataFrame:
        """Extract synchronized robot telemetry channels.
        
        Args:
            episode_path: Path to the episode directory.
            
        Returns:
            DataFrame containing 'timestamp' column and telemetry values.
        """
        pass


class ParserPluginRegistry:
    """Registry maintaining available episode parser plugins with priority resolution."""

    def __init__(self) -> None:
        self._plugins: Dict[str, ParserPlugin] = {}

    def register(self, plugin: ParserPlugin) -> None:
        """Register a parser plugin.
        
        Args:
            plugin: ParserPlugin instance to register.
        """
        self._plugins[plugin.format_name] = plugin

    def unregister(self, format_name: str) -> None:
        """Unregister a plugin by format name."""
        self._plugins.pop(format_name, None)

    def get_plugin(self, format_name: str) -> Optional[ParserPlugin]:
        """Retrieve plugin by format name."""
        return self._plugins.get(format_name)

    def find_matching_plugin(self, episode_path: Path) -> Optional[ParserPlugin]:
        """Find the highest-priority plugin that matches the episode directory.
        
        Args:
            episode_path: Directory path of the episode.
            
        Returns:
            Matching ParserPlugin with highest priority, or None if none match.
        """
        matches = [
            plugin for plugin in self._plugins.values()
            if plugin.matches_format(episode_path)
        ]
        if not matches:
            return None
        matches.sort(key=lambda p: p.priority, reverse=True)
        return matches[0]

    def list_plugins(self) -> List[str]:
        """List all registered format names."""
        return sorted(list(self._plugins.keys()))


# Global default registry
default_registry = ParserPluginRegistry()
