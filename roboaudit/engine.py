"""Unified end-to-end RoboAudit Engine orchestrator.

Integrates:
- Episode ingestion via EpisodeReader
- Timebase jitter and desync checks
- Multi-camera synchronization and stream swap detection
- Sensor dropout, freeze, and outlier monitoring
- Semantic action phase segmentation and timeline generation
- Gripper-vision contradiction analysis
- Operator mistake classification
- Mathematical invariant verification
- Goal alignment scoring
- AuditReport generation and schema validation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from roboaudit.analysis.alignment import GoalAlignmentScorer
from roboaudit.analysis.camera_sync import CameraSyncChecker
from roboaudit.analysis.dropout import SensorDropoutMonitor
from roboaudit.analysis.grasp import GraspAnomalyDetector
from roboaudit.analysis.invariants import InvariantChecker
from roboaudit.analysis.operator_mistakes import OperatorMistakeClassifier
from roboaudit.analysis.segmenter import ActionPhaseSegmenter
from roboaudit.analysis.timebase import TimebaseVerifier
from roboaudit.core.config import load_config
from roboaudit.core.models import (
    AuditConfig,
    AuditReport,
    DataIssue,
    EpisodeContext,
    GoalAlignment,
    OperatorMistake,
    TaskCompletion,
    TaskOutcome,
    Timeline,
)
from roboaudit.parsers.reader import EpisodeReader
from roboaudit.reporting.generator import AuditReportGenerator


class RoboAuditEngine:
    """Unified engine executing end-to-end quality and invariant audits on robotics episodes."""

    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize engine with audit configuration.
        
        Args:
            config: Optional audit configuration thresholds.
        """
        self.config = config or load_config()
        self.reader = EpisodeReader()
        self.timebase_verifier = TimebaseVerifier(self.config)
        self.camera_checker = CameraSyncChecker(self.config)
        self.dropout_monitor = SensorDropoutMonitor(self.config)
        self.segmenter = ActionPhaseSegmenter(self.config)
        self.grasp_detector = GraspAnomalyDetector(self.config)
        self.mistake_classifier = OperatorMistakeClassifier(self.config)
        self.invariant_checker = InvariantChecker()
        self.alignment_scorer = GoalAlignmentScorer()
        self.report_generator = AuditReportGenerator()

    def audit_episode(
        self,
        episode_path: Union[str, Path],
        config: Optional[AuditConfig] = None,
    ) -> AuditReport:
        """Run full deterministic audit on a single episode.
        
        Args:
            episode_path: Path to directory or archive containing episode data.
            config: Optional configuration override.
            
        Returns:
            Validated AuditReport instance.
        """
        cfg = config or self.config
        path = Path(episode_path)

        # 1. Ingest episode
        episode = self.reader.read(path)

        data_issues: List[DataIssue] = []

        # 2. Timebase analysis
        # Telemetry timebase
        if not episode.telemetry.empty:
            telem_issues, _ = self.timebase_verifier.check_telemetry_timebase(
                episode.telemetry,
                expected_rate_hz=30.0,
            )
            for ti in telem_issues:
                data_issues.append(ti.to_data_issue())

        # Video streams timebase
        for cam_id, frames in episode.videos.items():
            if frames:
                vid_issues, _ = self.timebase_verifier.check_video_timebase(
                    frames,
                    expected_fps=30.0,
                )
                for vi in vid_issues:
                    data_issues.append(vi.to_data_issue())

        # 3. Multi-camera synchronization & stream swap
        if len(episode.videos) >= 2:
            cam_issues = self.camera_checker.check_all(
                episode.videos,
                tolerance_ms=cfg.camera_sync_tolerance_ms,
            )
            for ci in cam_issues:
                data_issues.append(ci.to_data_issue())

        # 4. Sensor dropouts, freezes, and outliers
        if not episode.telemetry.empty:
            dropout_issues = self.dropout_monitor.monitor_all(episode.telemetry)
            for di in dropout_issues:
                data_issues.append(di.to_data_issue())

        # 5. Temporal segmentation into Action Phases
        timeline = self.segmenter.segment_episode(
            episode.telemetry,
            video_streams=episode.videos,
            config=cfg,
        )

        # 6. Grasp anomalies
        grasp_anomalies = self.grasp_detector.detect_anomalies(
            episode.telemetry,
            video_streams=episode.videos,
            config=cfg,
        )
        for ga in grasp_anomalies:
            data_issues.append(ga.to_data_issue())

        # 7. Operator mistake classification
        operator_mistakes = self.mistake_classifier.classify_mistakes(
            telemetry=episode.telemetry,
            video_streams=episode.videos,
            timeline=timeline,
            grasp_anomalies=grasp_anomalies,
            config=cfg,
        )

        # 8. Infer or parse task completion
        outcome_str = episode.metadata.task_outcome or "success"
        task_outcome = TaskOutcome(
            outcome=outcome_str,
            goal_reached_at_s=round(episode.metadata.duration_s * 0.9, 2) if outcome_str == "success" else None,
            undone_at_s=round(episode.metadata.duration_s * 0.95, 2) if outcome_str == "success_then_undone" else None,
            completed_at_s=round(episode.metadata.duration_s * 0.9, 2) if outcome_str == "success" else None,
        )

        task_completion = TaskCompletion(
            task_completed=outcome_str in {"success", "success_then_undone"},
            goal_reached_at_s=task_outcome.goal_reached_at_s,
            undone_at_s=task_outcome.undone_at_s,
            completed_at_s=task_outcome.completed_at_s,
            reason=f"Outcome classified as {outcome_str}",
        )

        # Context
        context = EpisodeContext(
            dataset=episode.metadata.dataset,
            rig=episode.metadata.robot_type,
            length_s=round(episode.metadata.duration_s, 2),
            instruction=episode.metadata.instruction,
            episode_id=episode.metadata.episode_id,
        )

        # 9. Verify invariants
        violations = self.invariant_checker.verify_all_invariants(
            outcome=task_outcome,
            timeline=timeline,
            episode_duration_s=episode.metadata.duration_s,
            alignment_relation="aligned",
        )
        for v in violations:
            data_issues.append(
                DataIssue(
                    issue=f"invariant_violation: {v.invariant_name} - {v.details}",
                    category="timebase" if "time" in v.invariant_name else "sensor",
                    severity="high" if v.severity == "error" else "medium",
                    t_s=0.0,
                    evidence=[f"rule_{v.invariant_name}"],
                )
            )

        # 10. Goal alignment scoring
        align_res = self.alignment_scorer.calculate_goal_alignment(
            outcome=outcome_str,
            timeline=timeline,
            mistakes=operator_mistakes,
            issues=data_issues,
        )
        goal_alignment = align_res.to_goal_alignment()

        # 11. Emitting and validating final structured AuditReport
        report = self.report_generator.generate_report(
            context=context,
            timeline=timeline,
            completion=task_completion,
            goal_alignment=goal_alignment,
            data_issues=data_issues,
            operator_mistakes=operator_mistakes,
        )

        return report
