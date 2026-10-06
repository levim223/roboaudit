"""Batch Episode Processing module with parallel execution and aggregate reporting.

Implements:
- Batch processing of multiple episode directories (Req 13.1)
- Parallel worker thread pool with configurable concurrency (Req 13.2)
- Individual audit report emission for each episode (Req 13.3)
- Aggregate summary report with cross-episode statistics (Req 13.4)
- Error resilience: graceful continuation on individual failures (Req 13.5)
- Real-time progress reporting (Req 13.6)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Union

from roboaudit.core.models import AuditConfig, AuditReport
from roboaudit.engine import RoboAuditEngine
from roboaudit.reporting.generator import AuditReportGenerator


@dataclass
class SingleEpisodeResult:
    """Audit result for an individual episode in a batch.
    
    Attributes:
        episode_path: Path to episode
        success: Whether the audit completed successfully
        report: Optional AuditReport if successful
        error: Optional error message if failed
        duration_s: Processing duration in seconds
    """
    episode_path: Path
    success: bool
    report: Optional[AuditReport] = None
    error: Optional[str] = None
    duration_s: float = 0.0


@dataclass
class BatchSummary:
    """Aggregate quality and health statistics across a batch of audited episodes.
    
    Attributes:
        total_episodes: Total number of episodes queued
        successful_audits: Number of successfully audited episodes
        failed_audits: Number of episodes that failed audit execution
        mean_quality_score: Mean quality score across all successful episodes
        total_anomalies: Sum of all detected anomalies across batch
        total_critical_issues: Sum of all critical issues across batch
        pass_rate: Percentage of episodes meeting quality criteria (e.g. quality_score >= 0.70)
        episode_scores: Mapping from episode_id to quality_score
        failed_episodes: List of failure details for unparsed episodes
    """
    total_episodes: int
    successful_audits: int
    failed_audits: int
    mean_quality_score: float
    total_anomalies: int
    total_critical_issues: int
    pass_rate: float
    episode_scores: Dict[str, float] = field(default_factory=dict)
    failed_episodes: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert BatchSummary to serializable dictionary."""
        return {
            "total_episodes": self.total_episodes,
            "successful_audits": self.successful_audits,
            "failed_audits": self.failed_audits,
            "mean_quality_score": round(self.mean_quality_score, 4),
            "total_anomalies": self.total_anomalies,
            "total_critical_issues": self.total_critical_issues,
            "pass_rate": round(self.pass_rate, 2),
            "episode_scores": self.episode_scores,
            "failed_episodes": self.failed_episodes,
        }

    def to_markdown(self) -> str:
        """Format summary as a clean markdown report."""
        lines = [
            "# RoboAudit Batch Summary Report",
            "",
            f"- **Total Episodes:** {self.total_episodes}",
            f"- **Successful Audits:** {self.successful_audits}",
            f"- **Failed Audits:** {self.failed_audits}",
            f"- **Pass Rate:** {self.pass_rate:.1f}%",
            f"- **Mean Quality Score:** {self.mean_quality_score:.3f}",
            f"- **Total Anomalies Detected:** {self.total_anomalies}",
            f"- **Critical Issues:** {self.total_critical_issues}",
            "",
            "## Episode Rankings",
            "| Episode ID | Quality Score | Status |",
            "| :--- | :--- | :--- |",
        ]
        for ep_id, score in sorted(self.episode_scores.items(), key=lambda x: x[1], reverse=True):
            status = "PASS" if score >= 0.70 else "FAIL"
            lines.append(f"| `{ep_id}` | {score:.3f} | **{status}** |")

        if self.failed_episodes:
            lines.append("")
            lines.append("## Processing Failures")
            for f in self.failed_episodes:
                lines.append(f"- **{f['episode_path']}**: {f['error']}")

        return "\n".join(lines)


class BatchProcessor:
    """Executes concurrent audits across datasets with error resilience and summary reporting."""

    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize batch processor with audit configuration.
        
        Args:
            config: Optional audit configuration.
        """
        self.config = config
        self.engine = RoboAuditEngine(config)
        self.report_generator = AuditReportGenerator()

    def process_batch(
        self,
        episode_paths: List[Union[str, Path]],
        max_workers: int = 4,
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[List[SingleEpisodeResult], BatchSummary]:
        """Audit multiple episodes concurrently.
        
        Requirements 13.1, 13.2, 13.3, 13.5, 13.6.
        
        Args:
            episode_paths: List of episode folder or archive paths.
            max_workers: Thread concurrency level.
            output_dir: Optional directory to save individual reports.
            progress_callback: Optional callback receiving (completed, total, current_item).
            
        Returns:
            Tuple of (individual results list, BatchSummary).
        """
        paths = [Path(p) for p in episode_paths]
        total = len(paths)
        results: List[SingleEpisodeResult] = []
        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        if total == 0:
            empty_summary = self.generate_summary([])
            return results, empty_summary

        def _process_single(ep_path: Path) -> SingleEpisodeResult:
            t0 = time.time()
            try:
                report = self.engine.audit_episode(ep_path, config=self.config)
                dur = time.time() - t0

                if out_path:
                    # Write individual JSON & Markdown reports (Req 13.3)
                    ep_id = report.context.episode_id
                    json_file = out_path / f"{ep_id}_report.json"
                    md_file = out_path / f"{ep_id}_report.md"
                    json_file.write_text(self.report_generator.export_json(report), encoding="utf-8")
                    md_file.write_text(self.report_generator.export_markdown(report), encoding="utf-8")

                return SingleEpisodeResult(
                    episode_path=ep_path,
                    success=True,
                    report=report,
                    error=None,
                    duration_s=round(dur, 3),
                )
            except Exception as e:
                dur = time.time() - t0
                return SingleEpisodeResult(
                    episode_path=ep_path,
                    success=False,
                    report=None,
                    error=str(e),
                    duration_s=round(dur, 3),
                )

        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(_process_single, p): p for p in paths}
            for future in as_completed(future_to_path):
                res = future.result()
                results.append(res)
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total, str(res.episode_path.name))

        summary = self.generate_summary(results)

        if out_path:
            summary_json = out_path / "batch_summary.json"
            summary_md = out_path / "batch_summary.md"
            summary_json.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
            summary_md.write_text(summary.to_markdown(), encoding="utf-8")

        return results, summary

    def generate_summary(
        self,
        results: List[SingleEpisodeResult],
    ) -> BatchSummary:
        """Calculate quality statistics across all audit results.
        
        Requirements 13.4, 13.5: Generates aggregate statistics across episodes,
        continuing when individual episodes fail.
        
        Args:
            results: List of SingleEpisodeResult instances.
            
        Returns:
            BatchSummary instance.
        """
        total = len(results)
        successful = [r for r in results if r.success and r.report is not None]
        failed = [r for r in results if not r.success]

        total_anomalies = 0
        total_critical = 0
        scores: Dict[str, float] = {}
        passing_count = 0

        for r in successful:
            rep = r.report
            assert rep is not None
            total_anomalies += rep.quality_metrics.total_anomalies
            total_critical += rep.quality_metrics.critical_issues
            q_score = rep.quality_metrics.quality_score
            scores[rep.context.episode_id] = round(q_score, 3)
            if q_score >= 0.70:
                passing_count += 1

        mean_score = (
            float(sum(scores.values()) / len(scores)) if scores else 0.0
        )
        pass_rate = (
            float(passing_count / len(successful) * 100.0) if successful else 0.0
        )

        failed_details = [
            {"episode_path": str(f.episode_path), "error": str(f.error)}
            for f in failed
        ]

        return BatchSummary(
            total_episodes=total,
            successful_audits=len(successful),
            failed_audits=len(failed),
            mean_quality_score=round(mean_score, 3),
            total_anomalies=total_anomalies,
            total_critical_issues=total_critical,
            pass_rate=round(pass_rate, 1),
            episode_scores=scores,
            failed_episodes=failed_details,
        )
