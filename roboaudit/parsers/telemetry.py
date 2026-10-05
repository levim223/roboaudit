"""Telemetry extraction and parsing for robotics demonstration logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
import pandas as pd


class TelemetryExtractor:
    """Extracts synchronized robot telemetry from CSV, JSON, or Parquet files."""

    SUPPORTED_EXTENSIONS = {".csv", ".json", ".parquet", ".parquet.gzip"}

    def extract(self, telemetry_path: Path) -> pd.DataFrame:
        """Extract telemetry data into a pandas DataFrame.
        
        Args:
            telemetry_path: Path to telemetry file.
            
        Returns:
            DataFrame containing time-series telemetry data with 'timestamp' column.
            
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is unsupported or contains no timestamp.
        """
        path = Path(telemetry_path)
        if not path.is_file():
            raise FileNotFoundError(f"Telemetry file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check if it has a records key or is column-oriented
                if "records" in data and isinstance(data["records"], list):
                    df = pd.DataFrame(data["records"])
                elif "data" in data and isinstance(data["data"], list):
                    df = pd.DataFrame(data["data"])
                else:
                    df = pd.DataFrame(data)
            else:
                raise ValueError("JSON telemetry must be an array or object of records")
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported telemetry format: {suffix}")

        df = self._standardize_timestamps(df)
        return df

    def _standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a standardized 'timestamp' column in seconds exists."""
        if "timestamp" in df.columns:
            return df
        elif "timestamp_us" in df.columns:
            df["timestamp"] = df["timestamp_us"] / 1_000_000.0
            return df
        elif "timestamp_ms" in df.columns:
            df["timestamp"] = df["timestamp_ms"] / 1_000.0
            return df
        elif "time_s" in df.columns:
            df["timestamp"] = df["time_s"]
            return df
        elif "t" in df.columns:
            df["timestamp"] = df["t"]
            return df

        # If no timestamp column found, infer 30Hz or index-based timestamps
        df["timestamp"] = df.index * (1.0 / 30.0)
        return df
