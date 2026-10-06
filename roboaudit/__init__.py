"""RoboAudit: Deterministic data quality engine for robotics learning demonstrations."""

from roboaudit.core.models import (
    AuditConfig,
    AuditReport,
    DataIssue,
    EpisodeContext,
    EpisodeMetadata,
    ExtractedEpisode,
    GoalAlignment,
    OperatorMistake,
    QualityMetrics,
    TaskCompletion,
    TaskOutcome,
    TemporalWindow,
    Timeline,
)
from roboaudit.analysis.timebase import TimebaseIssue
from roboaudit.core.config import (
    ROBOT_PROFILES,
    config_to_dict,
    load_config,
    validate_config_dict,
)
from roboaudit.engine import RoboAuditEngine
from roboaudit.batch import BatchProcessor, BatchSummary
from roboaudit.visualization.board import TimelineBoardGenerator, serve_board
from roboaudit.reporting.generator import AuditReportGenerator, AuditParser, AuditPrettyPrinter

__version__ = "1.0.0"

__all__ = [
    "AuditConfig",
    "AuditReport",
    "AuditReportGenerator",
    "AuditParser",
    "AuditPrettyPrinter",
    "BatchProcessor",
    "BatchSummary",
    "DataIssue",
    "EpisodeContext",
    "EpisodeMetadata",
    "ExtractedEpisode",
    "GoalAlignment",
    "OperatorMistake",
    "QualityMetrics",
    "ROBOT_PROFILES",
    "RoboAuditEngine",
    "TaskCompletion",
    "TaskOutcome",
    "TemporalWindow",
    "Timeline",
    "TimelineBoardGenerator",
    "TimebaseIssue",
    "config_to_dict",
    "load_config",
    "serve_board",
    "validate_config_dict",
]
