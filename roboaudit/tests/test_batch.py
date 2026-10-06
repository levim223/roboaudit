"""Integration tests for BatchProcessor and aggregate summary reporting.

Covers:
- Task 19.1: Parallel batch processing with worker threads (Req 13.1, 13.2)
- Task 19.2: Aggregate summary generation and failure resilience (Req 13.4, 13.5)
- Individual report persistence and progress tracking (Req 13.3, 13.6)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from roboaudit.batch import BatchProcessor, BatchSummary, SingleEpisodeResult


def create_mock_episode_dir(ep_dir: Path, ep_id: str, duration_s: float = 3.0) -> Path:
    """Helper to populate a valid standard episode directory."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    telem_dir = ep_dir / "telemetry"
    telem_dir.mkdir(exist_ok=True)

    # Telemetry CSV
    n_samples = int(duration_s * 30)
    timestamps = np.linspace(0.0, duration_s, n_samples)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "joint_0": np.sin(timestamps),
            "gripper_state": np.full(n_samples, 0.9),
            "velocity": np.full(n_samples, 0.1),
        }
    )
    df.to_csv(telem_dir / "joint_states.csv", index=False)

    # Metadata JSON
    metadata = {
        "episode_id": ep_id,
        "dataset": "batch_test_ds",
        "duration_s": duration_s,
        "instruction": f"Execute action for {ep_id}",
        "robot_type": "umi",
        "task_outcome": "success",
    }
    (ep_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return ep_dir


class TestBatchProcessor:
    """Test suite for batch episode processing."""

    def test_process_batch_multiple_episodes(self, tmp_path: Path):
        """Test Requirement 13.1, 13.2, 13.3: Concurrent processing of multiple episodes."""
        episodes_dir = tmp_path / "episodes"
        output_dir = tmp_path / "reports_out"

        ep1 = create_mock_episode_dir(episodes_dir / "ep_1", "ep_1")
        ep2 = create_mock_episode_dir(episodes_dir / "ep_2", "ep_2")
        ep3 = create_mock_episode_dir(episodes_dir / "ep_3", "ep_3")

        progress_log = []

        def on_progress(completed, total, name):
            progress_log.append((completed, total, name))

        processor = BatchProcessor()
        results, summary = processor.process_batch(
            episode_paths=[ep1, ep2, ep3],
            max_workers=3,
            output_dir=output_dir,
            progress_callback=on_progress,
        )

        assert len(results) == 3
        assert all(r.success for r in results)
        assert summary.total_episodes == 3
        assert summary.successful_audits == 3
        assert summary.failed_audits == 0
        assert summary.mean_quality_score > 0.0
        assert summary.pass_rate == 100.0

        # Verify progress callback was triggered
        assert len(progress_log) == 3
        assert progress_log[-1][0] == 3

        # Verify individual and summary files written (Req 13.3, 13.4)
        assert (output_dir / "ep_1_report.json").is_file()
        assert (output_dir / "ep_1_report.md").is_file()
        assert (output_dir / "batch_summary.json").is_file()
        assert (output_dir / "batch_summary.md").is_file()

    def test_error_resilience_with_failing_episode(self, tmp_path: Path):
        """Test Requirement 13.5: Engine continues processing when an individual episode fails."""
        episodes_dir = tmp_path / "episodes_resilience"
        ep_valid = create_mock_episode_dir(episodes_dir / "ep_good", "ep_good")
        ep_bad = episodes_dir / "ep_corrupted"
        ep_bad.mkdir(parents=True, exist_ok=True)
        # Empty folder without telemetry or metadata will fail parser

        processor = BatchProcessor()
        results, summary = processor.process_batch(
            episode_paths=[ep_valid, ep_bad],
            max_workers=2,
        )

        assert len(results) == 2
        assert summary.total_episodes == 2
        assert summary.successful_audits == 1
        assert summary.failed_audits == 1
        assert len(summary.failed_episodes) == 1
        assert "ep_corrupted" in summary.failed_episodes[0]["episode_path"]

    def test_batch_summary_markdown_and_dict(self):
        """Test formatting of BatchSummary object."""
        summary = BatchSummary(
            total_episodes=2,
            successful_audits=2,
            failed_audits=0,
            mean_quality_score=0.912,
            total_anomalies=1,
            total_critical_issues=0,
            pass_rate=100.0,
            episode_scores={"ep_a": 0.95, "ep_b": 0.88},
            failed_episodes=[],
        )
        data = summary.to_dict()
        assert data["total_episodes"] == 2
        assert data["mean_quality_score"] == 0.912

        md = summary.to_markdown()
        assert "# RoboAudit Batch Summary Report" in md
        assert "ep_a" in md
        assert "PASS" in md
